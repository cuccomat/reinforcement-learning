# Reinforcement Learning

The content of this repository comes from a module delivered by Imperial College London.
It was deepened the theory of both tabular reinforcement learning and deep reinforcement learning and their design, implementation, and evaluation using Python and Pytorch.

## Table of Contents
* [Tabular Reinforcement Learning](#tabular-reinforcement-learning)
    * [Dynamic Programming](#dynamic-programming)
    * [Monte-Carlo Learning](#monte-carlo-learning)
    * [Temporal Difference Learning](#temporal-difference-learning)

## Tabular Reinforcement Learning

It is presented a Maze environment and its resolution through a Markov Decision Process.

### Dynamic Programming

The chosen algorithm is called value iteration. Differently to policy iteration this algorithm combines policy improvement that guarantees the convergence to a better solution and truncated policy evaluation steps (exploiting the Bellman equation as an update rule). Therefore, I decided to pick this method since it succeeds in achieving the same result of policy iteration with a reduced computational cost. 
Dynamic programming models, including value iteration, are based on the strong assumption of perfect model, and the complete knowledge of the environment.

Despite the convergence being achieved with a infinite number of iterations, we assume as reasonable stopping criterion a threshold $\theta$ = 0.0001, that assures a marginal error. 

### Monte-Carlo Learning

To solve this question, I used the on-policy first-visit method among the MC algorithms. This method, with respect to the every-police one, updates and improves the same policy used to make decisions, unifying the estimation of the values and the subsequent control. I decided to choose this one because it represents a compromise between learning action values for a near-optimal policy and exploring at the same time the environment. Therefore, even if the off-policy methods are in general more powerful than the one picked, I considered crucial for this problem the speed of convergence that is in general higher for on-policy methods. I fixed the initial policy considering the same probability for all the actions (0.25 for each of the 4 actions). I assumed to consider 1000 episodes as stopping criterion because it does not imply a high computational cost even if the algorithm converges approximately at the optimal values with less episodes. In order to estimate the value of the parameters $\epsilon$ related to the exploration/exploitation trade-off and $\alpha$ that defines the way the $Q$ value estimation is updated, I took into account numerous combined solutions. The chosen value of $\epsilon$ and $\alpha$ are respectively $\epsilon = e^{\frac{-8x}{n}}$ and $\alpha = e^{\frac{-20x}{n}}$. The relative appropriate explanation of such choices are discussed in \ref{sec:eps_and_alpha}.

### Temporal Difference Learning

The chosen method among the TD ones is on-policy SARSA. The reason why I chose this method is because it generally better in its online updating performance with respect to Q learning, that on the other hand converges faster. As previously discussed for the MC learner here we do not suppose the knowledge of the environment as long as I used the same initial policy and reasonable stopping criterion of the previous method. As regards the choice of $\epsilon$ and $\gamma$ I chose $\epsilon = e^{\frac{-10x}{n}}$ and $\alpha = 0.7$. 

## Deep Reinforcement Learning



