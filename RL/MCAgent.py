import minihack_env
from commons import *
from collections import defaultdict


class MCAgent():

    def __init__(self, id, eps = 0.01, gamma = 0.9):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id           = id
        self.action_space = minihack_env.ACTIONS
        self.learning     = True
        self.eps          = eps
        self.gamma        = gamma
        
        # Internal state: has Q (default 0) and returns (default: empty list)
        self.memory = defaultdict([-999, []])
        
        self.high_probability = 1 - self.eps + self.eps/len(self.action_space)
        self.low_probability  = self.eps/len(self.action_space)
        
        
    def build_probabilities(self, index):
        probabilities = np.array([self.low_probability for action in self.action_spaces])
        probabilities[index] = self.high_probability
        
        return probabilities
        
        
    def policy(self, state):
        """
        Get action in current state according to epsilon greedy policy
        """
        # Check Q value for every action:
        q_values = np.zeros(len(self.action_space))
        for i, action in enumerate(self.action_space):
            q_values[i] = self.memory[(state, action)][0]
        # Get the greedy action, turn into probabilities:
        best_index = np.argmax(q_values)
        probabilities = self.build_probabilities(best_index)
        
        # Generate random action according to eps-greedy policy
        return np.random.choice(self.action_space, p=probabilities)
            

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        
        return self.policy(state)
        

    def onEpisodeEnd(self, states_list, actions_list, rewards_list, episode):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward_list: Rewards collected during episode
        :param episode: the episode number
        :return:
        """
        
        # Initialize G
        g_value = 0
        # Reverse lists: go from T to 0
        actions_list = actions_list[::-1]
        states_list  = states_list[::-1]
        rewards_list = rewards_list[::-1]
        # Build state action pairs:
        state_action_pairs = []
        for t in range(1, len(actions_list)):
            
            
        # Go over rewards, from T-1 to 0
        for t in range(1, len(actions_list)):
            next_reward = rewards_list[t-1] # R_{t+1}
            g_value = self.gamma * g_value + next_reward
            # Check if state action appears in list
            
        
        


