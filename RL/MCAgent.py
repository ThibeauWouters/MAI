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

    def __init__(self, id, action_space=np.array([0,1,2,3], dtype=int), alpha=0.1, eps = 0.05, gamma = 1.0, 
                 max_episode_steps=50, eps_period = None):
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
        
        self.train_counter = 0
        if eps_period is None:
            eps_period = max_episode_steps // 2
        self.eps_period = eps_period
        print(f"MC scheduler: eps_period: {eps_period}")
        # ^ count amount of training, to adapt eps


        # Memory has Q-values and n, the number of updates, to compute an incremental average
        self.Q = defaultdict(default_memory_value_callable)
        
        
    def policy(self, state):
            """
            Get action in current state according to epsilon greedy policy
            """
            
            Q_of_actions = np.array([self.Q[(state, action)][0] for action in self.action_space])
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
        # Initialize G
        g_value = 0
        
        # Go over lists rewards, from T-1 to 0
        for t in range(iteration_counter - 2, -1, -1):
            
            # G <- gamma G + R_{t+1}
            g_value = self.gamma * g_value + self.rewards_list[t+1]
            # Get this timestep's state and action
            state  = self.states_list[t]  # St
            action = self.actions_list[t]  # At
            
            # Do the update rule
            if not appears_earlier(state, action, self.states_list, self.actions_list, t):
                # Get previous value of this (S, A) pair
                prev_avg, n = self.Q[(state, action)]
                # Do an incremental avg and store as new value:
                new_avg = incremental_avg(prev_avg, g_value, n)
                self.Q[(state, action)] = np.array([new_avg, n+1])
                
        # Eps scheduler: check if we have to adapt eps
        self.train_counter += 1
        if self.train_counter % self.eps_period == 0:
            # Change exploration rate 
            self.eps *= 0.5
            print(f"Changing eps to {self.eps}")
            
                
    def onIterationEnd(self, iteration_counter, next_state):
        pass