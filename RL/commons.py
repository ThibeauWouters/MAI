from typing import List
import numpy as np
import gym
import minihack_env
import minihack
from nle import nethack
from minihack import RewardManager
from tqdm import tqdm
import matplotlib.pyplot as plt

# Abstract classes, provided by the assignment

class AbstractAgent():

    def __init__(self, id, action_space):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = action_space

        # Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        # You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)
        self.learning = True

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


    def onEpisodeEnd(self, reward, episode):
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


# Our classes, which extend them:

class RandomAgent():

    def __init__(self, id, action_space):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = action_space
        self.learning = True

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """

        # Choose random action
        return np.random.choice(self.action_space)
        

    def onEpisodeEnd(self, reward, episode):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass
    
class FixedAgent():

    def __init__(self, id):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = minihack_env.ACTIONS
        self.learning = True
        # First, keep on going down, then, keep on going left
        self.first_action = 2
        self.second_action = 1 
        self.fixed_action = self.first_action 
        self.turned = False

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        
        # If we are still following first direction, check if we should turn
        if not self.turned:
            # Else, check if we can still do first action, otherwise turn
            observation = get_crop_chars_from_observation(state)
            position = np.argwhere(observation == 64)[0]
            x, y = position
            n, m = observation.shape
            next_x, next_y = get_next_grid_position(x, y, self.fixed_action, n, m)
            if observation[next_x, next_y] != 46:
                # Turn the agent
                self.fixed_action = self.second_action
                self.turned = True
        
        return self.fixed_action
        

    def onEpisodeEnd(self, reward, episode):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass


class RLTask():

    def __init__(self, env: gym.Env, agent: AbstractAgent):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env = env
        self.agent = agent

    def interact(self, n_episodes: int) -> float:
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode average returns  (see assignment) for a definition
        """
        
        # Initialize an empty list for the returns
        returns_list = np.zeros(n_episodes)
        avg_returns_list = np.zeros(n_episodes)
        
        # Do the episodes
        for i in tqdm(range(n_episodes)):
            # Reset everything
            state = self.env.reset()
            # TODO - incremental way of doing averages?
            return_value = 0
            rewards_list = []
            done = False
            reward = 0
            self.action_list = []
            
            # Do an episode:
            while not done:
                # Agent chooses interaction and interacts with environment
                action = self.agent.act(state, reward)
                state, reward, done, info = self.env.step(action)
                # state, reward = observation
                rewards_list.append(reward)
                # Save the actions for plotting
                self.action_list.append(action)
                
            # Episode is over, compute return
            return_value = np.sum(rewards_list)
            # Append the return to the list of returns
            returns_list[i] = return_value
            # Compute the average of the first episodes
            avg_returns_list[i] = np.mean(returns_list[:i+1])
            
        return avg_returns_list

    def visualize_episode(self, max_number_steps: int = 10, 
                          plot=False, name="visualization", title=""):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """
        
        # Reset environment
        fsize = 22  # font size for the title
        state = self.env.reset()
        print("=== Starting state: === \n")
        self.env.render()
        if plot:
            plt.imshow(get_crop_pixel_from_observation(state))
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Start", fontsize = fsize)
            plt.savefig(f"{name}_{0}.png", bbox_inches='tight')
            plt.close()
        print("\n")
        
        # Reconstruct the episode from the actions and information of the environment
        for i, action in enumerate(self.action_list):
            # Make a step in the environment
            state, _, done, _ = self.env.step(action)
            # Translate this action into readable language
            action = minihack_env.translate_action(action)
            # Render the environment to the screen:
            print(f"=== State {i+1}: (action = {action}) === \n")
            self.env.render()
            print("\n")
            # Save pixel plots to .png if desired
            if plot and not done:
                plt.imshow(get_crop_pixel_from_observation(state))
                plt.xticks([])
                plt.yticks([])
                plt.title(f"Iteration {i+1}", fontsize = fsize)
                plt.savefig(f"{name}_{i+1}.png", bbox_inches='tight')
                plt.close()
            # Stop rendering after specified max number steps is reached
            if i+1 == max_number_steps:
                return
            
            

def get_next_grid_position(x, y, action, n, m):
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

class GridWorld(gym.Env):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.action_space = [0, 1, 2, 3]
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(n), gym.spaces.Discrete(m)))
        self.reward_range = (-1, 0)
        self.position = (0, 0)
        self.goal = (n-1, m-1)
        
    def step(self, action):
        
        # The current state
        x, y = self.position
        x, y = get_next_grid_position(x, y, action, self.n, self.m)
            
        # Give rewards after action was performed
        if (x, y) == self.goal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        self.position = (x, y)
        return self.position, reward, done, {}

    def reset(self):
        self.position = (0, 0)
        return self.position
    
    def render(self):
        # Get an empty grid of specified size (n, m)
        grid = np.zeros((self.n, self.m), dtype=np.int8)
        grid[self.goal] = 2
        grid[self.position] = 1

        output = ''
        for row in grid:
            for val in row:
                if val == 0:
                    output += '.'
                elif val == 1:
                    output += 'X'
                elif val == 2:
                    output += 'G'
            output += '\n'
        print(output)