import numpy as np

class MC_agent(object):

    def solve(self, env):
        """
        Solve a given Maze environment using Monte Carlo learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """
        # Initialisation
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        policy = np.ones((env.get_state_size(), env.get_action_size())) * 0.25 
        values = [] 
        returns = {}
        
        num_episodes = 1000
        total_rewards = []
        
        # Iterate across the espisodes
        for epis in range(num_episodes):
            epsilon = np.exp(-8*epis/num_episodes) 
   
            alpha = np.exp(-20*epis/num_episodes)

            # Reset values
            G = 0.0 
            V = np.zeros(env.get_state_size())
            t, state, reward, done = env.reset()

            # Generate episode
            episode = []
            while done is False:
                action = np.random.choice(range(env.get_action_size()), p = policy[state, :])
                t, state_t1, reward_t1, done = env.step(action)
                episode.append((state, action, reward_t1))
                state = state_t1

            total_reward = 0.0
            for step in list(reversed(range(len(episode)))):
                state_t, action_t, reward_t1 = episode[step]
                G = reward_t1 + env.get_gamma() * G
                state_action = (state_t, action_t)
                total_reward += reward_t1

                # Verify that there is not the same state before (first visit)
                if not state_action in [(e[0], e[1]) for e in episode[0:step]]: 
                    if not state_action in returns:
                        returns[(state_action)] = [G]
                    else:    
                        returns[(state_action)].append(G)

                    # Update the state-action function
                    Q[state_t][action_t] = Q[state_t][action_t] +  alpha * (G - Q[state_t][action_t]) 

                    # Update the epsilon-greedy policy
                    A = np.argmax(Q[state_t, :])   
                    for action in range(env.get_action_size()): 
                        if action == A:
                            policy[state_t, action] = 1 - epsilon + epsilon / env.get_action_size() 
                        else:
                            policy[state_t, action] = epsilon / env.get_action_size()

            total_rewards.append(total_reward)
            # Compute the value for each episode and append in values
            for st in range(env.get_state_size()):                
                for action in range(env.get_action_size()):
                    V[st] += policy[st, action] * Q[st, action]
            values.append(V)

        return policy, values, total_rewards 