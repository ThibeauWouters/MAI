import minihack_env
import numpy as np
from commons import *
from commons import AbstractAgent
from collections import defaultdict
import pickle

VERBOSE = False

default_memory_value = np.array([0, 0])

def default_memory_value_callable():
    return default_memory_value


class MCAgent(AbstractAgent):

    def __init__(self, id, save_name, action_space=np.array([0,1,2,3], dtype=int), eps = 0.01, gamma = 1.0, 
                 max_episode_steps=50, load_name=""):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        
        super().__init__(id, action_space=action_space, max_episode_steps=max_episode_steps)
        
        self.learning = True
        self.eps      = eps
        self.gamma    = gamma
        
        self.high_probability = 1 - self.eps + self.eps/len(self.action_space)


        # Memory has Q-values and n, the number of updates, to compute an incremental average
        # Either create empty memory or load from previous training
        if load_name=="":
            self.memory = defaultdict(default_memory_value_callable)
        else:
            self.load_memory(load_name)
            
        # For saving the agents' policy:
        self.save_name = save_name + "memory.pkl"
            
        
        
    def policy(self, state):
            """
            Get action in current state according to epsilon greedy policy
            """
            
            Q_of_actions = np.array([self.memory[(state, action)][0] for action in self.action_space])
            # Check where Q value is maximal, can be multiple, so sample arbitrarily
            argmax_indices = np.argwhere(Q_of_actions == np.max(Q_of_actions)).flatten()
            argmax_index = np.random.choice(argmax_indices)
            
            if np.random.rand() < self.high_probability:
                action = argmax_index
            else:
                action = np.random.choice(self.action_space[self.action_space != argmax_index])
                
            return action
        
    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        
        # Convert state to hashed state
        state = np_hash(state)
        
        action = self.policy(state)
            
        return action
        
    def onEpisodeEnd(self, iteration_counter):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward_list: Rewards collected during episode
        :param episode: the episode number
        :return:
        """
        
        if VERBOSE:
            print(self.memory)
        
        # Initialize G
        g_value = 0
        
        # Limit the lists:
        self.actions_list = self.actions_list[:iteration_counter]
        self.states_list  = self.states_list[:iteration_counter]
        self.rewards_list = self.rewards_list[:iteration_counter]
        
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
                prev_avg, n = self.memory[(state, action)]
                # Do an incremental avg and store as new value:
                new_avg = incremental_avg(prev_avg, g_value, n)
                self.memory[(state, action)] = np.array([new_avg, n+1])
                
    def onIterationEnd(self, iteration_counter, next_state):
        pass
    
    def save_memory(self):
        
        # Open a file and use dump()
        with open(self.save_name, 'wb') as file:
            pickle.dump(self.memory, file)
            
    def load_memory(self, load_name):
        load_name += "memory.pkl"
        
        # Open a file and use dump()
        with open(load_name, 'rb') as file:
            self.memory = pickle.load(file)