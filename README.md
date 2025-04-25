# Rank-Based Bayesian Optimization

This repository contains the necessary updates on my ongoing thesis on the above topic. 
### Setup:
- **Configuration space of hyperparameters**: $\Theta$
- **(Random) metric function**: $S: \Theta \to \mathbb{R}$
- We can evaluate $S$ at points $\theta \in \Theta$, but each evaluation is costly.
- We assume distinct evaluations $S(\theta_1), S(\theta_2), \dots, S(\theta_n)$ are independent.
- We only have access to the order of the values: 
  $$
  S(\theta_{i1}) < S(\theta_{i2}) < \dots < S(\theta_{in})
  $$

### Goal:
Find hyperparameters $\theta^* \in \Theta$ which minimize $S$, while limiting the number of evaluations of $S$.


PL_Model Notebook: This notebook contains an inference model based on pairwise ranking data. The implementation uses Maximum Likelihood Estimation (MLE) to estimate the relative worth of different items based on their pairwise competition outcomes. Two different approaches were explored: The closed form formula and Gradient ascent with pytorch.
