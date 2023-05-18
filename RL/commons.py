from typing import List
import numpy as np
import gym
import minihack_env
import minihack
from nle import nethack
from minihack import RewardManager
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Callable

# Abstract classes, provided by the assignment

class AbstractAgent():

    def __init__(self, id, action_space=np.array([0,1,2,3])):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = action_space
        
        # Lists for observed states, actions, rewards
        self.states_list  = []
        self.actions_list = []
        self.rewards_list = []

        # Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        # You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)
        self.learning = True
        
    def clear_lists(self):
        self.states_list  = []
        self.actions_list = []
        self.rewards_list = []

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        raise NotImplementedError()


    def onEpisodeEnd(self):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass



class AbstractRLTask():

    def __init__(self, env, agent):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env = env
        self.agent = agent

    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode avergae returns  (see assignment for a definition
        """
        raise NotImplementedError()


    def visualize_episode(self, max_number_steps = None):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """

        raise NotImplementedError()

#######################
# Auxiliary functions #
#######################

blank = 32
def get_crop_chars_from_observation(observation):
    chars = observation["chars"]
    coords = np.argwhere(chars != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    chars = chars[x_min:x_max + 1, y_min:y_max + 1]
    return chars


size_pixel = 16
def get_crop_pixel_from_observation(observation):
    coords = np.argwhere(observation["chars"] != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    non_empty_pixels = observation["pixel"][x_min * size_pixel : (x_max + 1) * size_pixel, y_min * size_pixel : (y_max + 1) * size_pixel]
    return non_empty_pixels


def get_next_grid_position(x: int, y: int, action: int, n: int, m: int) -> Tuple[int, int]:
    """
    Determines next grid position based on action in (n, m) grid world.
    """
    
    # Check which compass direction was chosen
    if action == 0:
        # Go north
        x = max(0, x-1)
    elif action == 1:
        # Go east
        y = min(m-1, y+1)
    elif action == 2:
        # Go south
        x = min(n-1, x+1)
    elif action == 3:
        # Go west
        y = max(0, y-1)
        
    return x, y

def have_common_element(arr1: np.array, arr2: np.array) -> bool:
    """
    Checks whether arr1 and arr2 have at leasts one common element
    """
    common_elements = np.intersect1d(arr1, arr2)
    return len(common_elements) > 0


def incremental_avg(prev_avg, new_val, n):
    if n == 0:
        return new_val
    else:
        return prev_avg + (new_val - prev_avg)/n