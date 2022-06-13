import numpy as np

class TD_agent(object):
    
    def solve(self, env):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """

        # Initialisation
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        policy = np.ones((env.get_state_size(), env.get_action_size()))*0.25
        values = []
        total_rewards = []
        num_episodes = 1000                          
        
        for epis in range(num_episodes):
            epsilon = np.exp(-10*epis/num_episodes) 
            alpha = 0.7
                
            V = np.zeros(env.get_state_size())
            total_reward = 0.0
            
            t, state, reward, done = env.reset()
            state_t = state   
            action_t = np.random.choice(range(env.get_action_size()), p = policy[state_t, :])
            
            while done is False:  
                t, state_t1, reward_t, done = env.step(action_t) 
                
                total_reward += reward_t
                action_t1 = np.random.choice(range(env.get_action_size()), p = policy[state_t1, :]) 
                
                Q[state_t, action_t] = Q[state_t, action_t] + alpha * (reward_t + env.get_gamma() * Q[state_t1, action_t1] - Q[state_t, action_t])  
                 
                A = np.argmax(Q[state_t, :])   
                    
                for action in range(env.get_action_size()):
                    if action == A:
                        policy[state_t, action] = 1 - epsilon + (epsilon / env.get_action_size())
                       
                    else:
                        policy[state_t, action] = epsilon / env.get_action_size()
                        
                state_t = state_t1
                action_t = action_t1   
                
            total_rewards.append(total_reward)    
            for st in range(env.get_state_size()):                
                for action in range(env.get_action_size()):
                    V[st] += policy[st, action] * Q[st, action]         
                
            values.append(V)

        return policy, values, total_rewards
