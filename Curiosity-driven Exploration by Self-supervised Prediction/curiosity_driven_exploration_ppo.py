import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
import tyro
import numpy as np
from tqdm import tqdm
import time
import random
import os
import matplotlib.pyplot as plt


@dataclass
class Args:
    cuda: bool = True
    total_timesteps: int = 100000
    learning_rate: float = 1e-4
    num_envs: int = 1
    num_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    beta: float = 0.2
    teta: float = 0.01
    batch_size: int = num_steps * num_envs
    minibatch_size: int = batch_size // num_minibatches
    num_iterations: int = total_timesteps // batch_size


class Agent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class ICM(nn.Module):
    def __init__(self, input_dim, action_dim, embedding_dim=288):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(embedding_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(embedding_dim+1, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def get_encoding(self, state):
        return self.encoder(state)

    def get_estimated_action(self, state_embed, next_state_embed):
        x = torch.cat((state_embed, next_state_embed), dim=2)
        return self.inverse_model(x)

    def get_estimated_next_state(self, state_embed, action):
        x = torch.cat((state_embed, action), dim=2)
        return self.forward_model(x)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    set_seed()

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print(device)

    env = gym.make("Acrobot-v1")
    try:
        input_dim = env.observation_space.n
        one_hot = True
    except:
        input_dim = env.observation_space.shape[0]
        one_hot = False
    output_dim = env.action_space.n
    action_dim = env.action_space.shape

    agent = Agent(input_dim, output_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)
    icm = ICM(input_dim, output_dim).to(device)
    icm_optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)
    intrinsic_rew = torch.zeros((1, 1))

    #  This is faster than using a RolloutBuffer since it stay in the GPU
    obs = torch.zeros((args.num_steps, args.num_envs) + (input_dim,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_dim).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs).to(device)
    if one_hot:
        next_obs = F.one_hot(next_obs, input_dim).to(torch.float32)
    next_done = False
    tot_reward = 0
    reward_buff = []
    loop = tqdm(range(args.num_iterations))
    avg_rew = 0
    start_time = time.time()

    for i in loop:
        for step in range(args.num_steps):
            if next_done:
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs).to(device)
                if one_hot:
                    next_obs = F.one_hot(next_obs, input_dim).to(torch.float32)
                reward_buff.append(tot_reward)
                tot_reward = 0
                avg_rew = np.mean(reward_buff[-10:])

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, termination, truncation, infos = env.step(action.cpu().item())
            next_obs = torch.tensor(next_obs).to(device)
            if one_hot:
                next_obs = F.one_hot(next_obs, input_dim).to(torch.float32)
            rewards[step] = reward
            tot_reward += reward

        step_per_second = int(global_step // (time.time() - start_time))
        loop.set_postfix_str(
            f"{step_per_second}step/s, avg_reward={avg_rew:.1f}, intrinsic_rew={intrinsic_rew.mean():.2f}, global_step={global_step}")

        # Here we calculate the intrinsic reward and update the ICM model
        icm_states = obs[:-1]
        icm_next_states = obs[1:]
        # Run the encoder and get s and s' embedded
        states_embedded = icm.get_encoding(icm_states)
        next_states_embedded = icm.get_encoding(icm_next_states)
        # Run the inverse model with s and s' embed, obtain a^, loss with a and update
        estimated_action = icm.get_estimated_action(states_embedded, next_states_embedded)
        criterion = nn.CrossEntropyLoss()
        inverse_loss = criterion(estimated_action.view(-1, output_dim), actions[:-1].view(-1).long())
        # Run the forward model with a and s embed and obtain s'^ embed, loss with s'embed and update
        estimated_next_states = icm.get_estimated_next_state(states_embedded, actions[:-1].unsqueeze(2))
        forward_loss = nn.MSELoss()(estimated_next_states, next_states_embedded)
        #  Sum up the losses and back-propagate
        tot_loss = (1-args.beta)*inverse_loss + args.beta*forward_loss
        icm_optimizer.zero_grad()
        tot_loss.backward()
        icm_optimizer.step()
        # s'^ embed difference with s'embed is the intrinsic reward
        intrinsic_rew = torch.norm(estimated_next_states.detach() - next_states_embedded.detach(), p=2, dim=2)
        rewards[:-1] += 1 * intrinsic_rew

        #  Rollout finished, advantage calculation
        with torch.no_grad():
            #  If we don't apply the reset logic at the beginning of the for loop, but we put at the end
            #  we risk to have here an initial state instead of the terminal one here in case we ended in the terminal
            next_value = agent.get_value(next_obs).reshape(1, -1)
            adv = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
                adv[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
            returns = adv + values

        #  Net update
        b_obs = obs.reshape((-1,) + (input_dim, ))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_dim)
        b_advantages = adv.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

    # Plot average reward over time
    plt.figure(figsize=(10, 5))
    plt.plot(reward_buff, label='Episode Reward')
    window = 10
    if len(reward_buff) >= window:
        avg_rewards = np.convolve(reward_buff, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(reward_buff)), avg_rewards, label=f'{window}-Episode Moving Average')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Average Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
