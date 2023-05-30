import minihack_env
import numpy as np
from commons import *
from commons import AbstractAgent
from collections import defaultdict
import pickle

VERBOSE = True

default_Q_value = 0.0

def default_Q_value_callable():
    return default_Q_value

class QAgent(AbstractAgent):

    def __init__(self, id, save_name, action_space=np.array([0,1,2,3]), alpha = 0.1, eps = 0.05, gamma = 1.0, 
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
        self.alpha    = alpha

        self.high_probability = 1 - self.eps + self.eps/len(self.action_space)
        
        self.load_name = load_name
        
        # Internal state: has Q (default 0) and returns (default: empty list)
        if load_name=="":
            self.Q = defaultdict(default_Q_value_callable)
        else:
            
            self.load_memory(load_name)
            
        self.save_name = save_name + "memory.pkl"
        
    def policy(self, state):
            """
            Get action in current state according to epsilon greedy policy
            """
            
            Q_of_actions = np.array([self.Q[(state, action)] for action in self.action_space])
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
    
    def onIterationEnd(self, iteration_counter, next_state):
        
        # Need at least one full "SARSA" to make an update, first iteration not possible
        if iteration_counter == 0:
            return
        else:
            # Get the state, action
            state       = self.states_list[iteration_counter]
            action      = self.actions_list[iteration_counter]
            reward      = self.rewards_list[iteration_counter]
            
            # Update Q
            previous_Q = self.Q[(state, action)]
            new_Q = previous_Q + self.alpha * (reward + self.gamma * np.max([self.Q[(next_state, a)] for a in self.action_space]) - previous_Q)
            self.Q[(state, action)] = new_Q
        
    def onEpisodeEnd(self, iteration_counter):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward_list: Rewards collected during episode
        :param episode: the episode number
        :return:
        """
        
        pass
    
    def save_memory(self):
        
        # Don't save if we are checking the polciy
        if len(self.load_name) > 0:
            print("Not saving memory")
            pass
        
        # Open a file and use dump()
        with open(self.save_name, 'wb') as file:
            pickle.dump(self.Q, file)
            
    def load_memory(self, load_name):
        load_name += "memory.pkl"
        
        # Open a file and use dump()
        print(f"Loading memory from {load_name}")
        with open(load_name, 'rb') as file:
            self.Q = pickle.load(file)
            print(self.Q)