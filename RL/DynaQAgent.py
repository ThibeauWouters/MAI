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

class DynaQAgent(AbstractAgent):

    def __init__(self, id, action_space=np.array([0,1,2,3]), alpha = 0.1, eps = 0.05, gamma = 1.0, n = 10,
                 max_episode_steps=50):
        
        super().__init__(id, action_space=action_space, max_episode_steps=max_episode_steps)
        
        self.learning = True
        self.eps      = eps
        self.gamma    = gamma
        self.alpha    = alpha

        self.Q = defaultdict(default_Q_value_callable)
        
        # This is new for the planning:
        self.model = {}
        self.n = n
        
    def policy(self, state):
            """
            Get action in current state according to epsilon greedy policy
            """
            
            high_probability = 1 - self.eps + self.eps/len(self.action_space)
            
            Q_of_actions = np.array([self.Q[(state, action)] for action in self.action_space])
            # Check where Q value is maximal, can be multiple, so sample arbitrarily
            argmax_indices = np.argwhere(Q_of_actions == np.max(Q_of_actions)).flatten()
            argmax_index = np.random.choice(argmax_indices)
            
            if np.random.rand() < high_probability:
                action = argmax_index
            else:
                action = np.random.choice(self.action_space[self.action_space != argmax_index])
                
            return action

    def act(self, state, reward=0):
        
        # Convert state to hashed state
        state = np_hash(state)
        action = self.policy(state)
            
        return action
    
    def onIterationEnd(self, iteration_counter, next_state):
        
        # Update Q values 
        state       = self.states_list[iteration_counter]
        action      = self.actions_list[iteration_counter]
        reward      = self.rewards_list[iteration_counter]
        
        previous_Q = self.Q[(state, action)]
        new_Q = previous_Q + self.alpha * (reward + self.gamma * np.max([self.Q[(next_state, a)] for a in self.action_space]) - previous_Q)
        self.Q[(state, action)] = new_Q
        
        # Save to model
        
        self.model[(state, action)] = [reward, next_state]
        
        for _ in range(self.n):
            # Sample key from model dict, is a state-action pair
            index = np.random.choice(len(self.model.keys()))
            state, action = list(self.model.keys())[index]
            # Get the reward and next state of this pair
            reward, next_state = self.model[(state, action)]
            # Update rule
            previous_Q = self.Q[(state, action)]
            new_Q = previous_Q + self.alpha * (reward + self.gamma * np.max([self.Q[(next_state, a)] for a in self.action_space]) - previous_Q)
            self.Q[(state, action)] = new_Q
        
    def onEpisodeEnd(self, iteration_counter):
        
        pass
    