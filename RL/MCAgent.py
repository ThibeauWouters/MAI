import minihack_env
from commons import *
from collections import defaultdict

VERBOSE = False

default_memory_value = [0, 0]
default_A_star_value = 0

def default_memory_value_callable():
    return default_memory_value

def default_A_star_value_callable():
    return default_A_star_value

def np_hash(arr):
    return hash(arr.data.tobytes())
    # return hash(arr.tostring())


class MCAgent(AbstractAgent):

    def __init__(self, id, action_space=np.array([0,1,2,3]), eps = 0.01, gamma = 1.0):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        
        super().__init__(id, action_space=np.array([0,1,2,3]))
        
        self.learning     = True
        self.eps          = eps
        self.gamma        = gamma
        
        # Internal state: has Q (default 0) and returns (default: empty list)
        self.memory = defaultdict(default_memory_value_callable)
        self.A_star = defaultdict(default_A_star_value_callable)
        
        self.high_probability = 1 - self.eps + self.eps/len(self.action_space)
        self.low_probability  = self.eps/len(self.action_space)
        
        
    def policy(self, state):
        """
        Get action in current state according to epsilon greedy policy
        """
        
        A_star = self.A_star[np_hash(state)]
        
        if np.random.rand() < self.high_probability:
            return A_star
        else:
            return np.random.choice(self.action_space[self.action_space != A_star])
        
            

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        
        a = self.policy(state)
            
        return a
        
    def onEpisodeEnd(self):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward_list: Rewards collected during episode
        :param episode: the episode number
        :return:
        """
        
        if VERBOSE:
            print(self.memory)
            # print(self.states_list)
        
        # Initialize G
        g_value = 0
        # Reverse lists: go from T to 0
        
        self.actions_list = np.array(self.actions_list[::-1])
        self.states_list  = np.array(self.states_list[::-1])
        self.rewards_list = np.array(self.rewards_list[::-1])
        
        # Go over lists rewards, from T-1 to 0
        for t in range(1, len(self.actions_list)):
            
            # G <- gamma G + R_{t+1}
            g_value = self.gamma * g_value + self.rewards_list[t-1]
            # Get this timestep's state and action
            state  = self.states_list[t]
            action = self.actions_list[t]
            
            # Check whether we have to update: certainly at end of array
            update = False  
            if t == len(self.actions_list - 1):
                update = True
            else:
                # If (S, A) reappears earlier in trajectory, we update later
                same_states_indices  = np.argwhere(self.states_list[t+1:]  == state).flatten()
                same_actions_indices = np.argwhere(self.actions_list[t+1:] == action).flatten()
                update = not(have_common_element(same_states_indices, same_actions_indices))
                
            # Do the update rule
            if update:
                # Get previous value of this (S, A) pair
                prev_avg, n = self.memory[(np_hash(state), action)]
                # Do an incremental avg and store as new value:
                new_avg = incremental_avg(prev_avg, g_value, n)
                self.memory[(np_hash(state), action)] = [new_avg, n+1]
                # Update A_star: argmax of the Q values viewed as function of actions
                Q_of_actions = np.array([self.memory[(np_hash(state), action)][0] for action in self.action_space])
                self.A_star[np_hash(state)] = np.argmax(Q_of_actions)
                
                
            
        
        


