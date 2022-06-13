import numpy as np

class DP_agent(object):

    def solve(self, env):
        """
        Solve a given Maze environment using Dynamic Programming
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - V {np.array} -- Corresponding value function 
        """

        # Initialisation (can be edited)
        epochs = 0
        policy = np.zeros((env.get_state_size(), env.get_action_size()))
        V = np.zeros(env.get_state_size())
        v = np.zeros(env.get_state_size())
        threshold = 0.0001
        delta = threshold
        
        # Ensure gamma value is valid
        assert (env.get_gamma() <= 1) and (
            env.get_gamma() >= 0), "Discount factor should be in [0, 1]."

        while delta >= threshold:
            delta = 0
            epochs += 1
            states = 0
            for prior_state in range(env.get_state_size()):
                
                if not env.get_absorbing()[0, prior_state]:
                    states += 1
                    v = V[prior_state]
                    Q = np.zeros(env.get_action_size())
                    for next_state in range(env.get_state_size()):
                        Q += env.get_T()[prior_state, next_state, :] * (env.get_R()[prior_state, next_state, :] + env.get_gamma() * V[next_state]) 
                    V[prior_state] = np.max(Q)                
                    delta = max(delta, np.abs(v - V[prior_state]))
                    action_max = np.argmax(Q)
                    policy[prior_state,:] = np.zeros(env.get_action_size())
                    policy[prior_state,action_max] = 1

        return policy, V
