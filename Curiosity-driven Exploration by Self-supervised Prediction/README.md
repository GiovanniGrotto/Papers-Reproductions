# 🧾 Paper: Curiosity-driven Exploration by Self-supervised Prediction

**Authors**: Deepak Pathak and Pulkit Agrawal and Alexei A. Efros and Trevor Darrell\
**Paper link**: https://proceedings.mlr.press/v70/pathak17a.html?ref=https://githubhelp.com

---

## 🧠 Summary

This paper proposes an intrinsic motivation mechanism for reinforcement learning agents based on curiosity. Instead of relying only on external rewards from the environment, the agent generates its own internal reward signal by measuring how well it can predict the consequences of its actions.

The key idea is to learn a forward dynamics model in a compact, learned feature space, and use the prediction error as a curiosity reward. This allows the agent to explore parts of the environment that are novel or hard to predict — effectively encouraging exploration without requiring any extrinsic reward.

In my implementation, I compared PPO with and without the intrinsic curiosity reward in the Acrobot and Taxi environments from Gymnasium. The version using intrinsic reward achieved better performance in both environments.

This is the learning curve for Acrobot: plain PPO on the left, and PPO with intrinsic reward on the right.
![Train](/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/imgs/Acrobot.png)

This is the learning curve for Taxi: plain PPO on the left, and PPO with intrinsic reward on the right.
![Train](/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/imgs/Taxi.png)

In both cases, the implementation with intrinsic reward reaches better results faster and avoids the flat reward plateau at the beginning of training.
