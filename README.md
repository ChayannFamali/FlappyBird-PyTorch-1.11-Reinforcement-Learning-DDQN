# FlappyBird PyTorch-1.11-Reinforcement-Learning DDQN

This project was taken as a basis https://github.com/PacktPublishing/PyTorch-1.x-Reinforcement-Learning-Cookbook/tree/master/Chapter09

The usual DQN tends to overestimate the Q-values of potential actions in a given state. This wouldn't cause any problems if all actions were equally overrated, but the fact is that once one particular action becomes overrated, it's more likely that it will be selected in the next iteration, making it difficult for the agent to uniformly explore the environment. and find the right policy.

The trick to overcoming this is quite simple. We are going to separate the action selection from the target generation of the Q-value. To do this, we will have two separate networks, so the name is Double Q-Learning. We will use our main network to select an action, and the target network to generate a Q value for that action. To synchronize our networks, we are going to copy weights from the primary network to the target one every (usually about 10k) training steps.


![image](https://user-images.githubusercontent.com/65254370/158546274-fd484a5f-6fb8-4e10-bfe9-0c31d450f3fa.png)
