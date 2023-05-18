import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gym 
from commons import *

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
        :return: a list of episode average returns (see assignment for a definition)
        """
        
        # Initialize an empty list for the returns
        returns_list = np.zeros(n_episodes)
        avg_returns_list = np.zeros(n_episodes)
        
        # Do the episodes
        for i in tqdm(range(n_episodes)):
            # Reset everything
            observation = self.env.reset()
            return_value = 0
            done = False
            reward = 0
            
            # Clear states, actions and rewards lists
            self.agent.clear_lists()
            
            # Do an episode:
            while not done:
                # Save current state
                # Convert state to the cropped representation
                state = get_crop_chars_from_observation(observation)
                self.agent.states_list.append(state.copy())
                # Agent chooses interaction and interacts with environment
                action = self.agent.act(state, reward)  
                observation, reward, done, _ = self.env.step(action)
                # Save chosen action and observed reward
                self.agent.actions_list.append(action)
                self.agent.rewards_list.append(reward)
            
            # Let the agent learn after end of episode:
            self.agent.onEpisodeEnd()
            
            # Episode is over, compute return
            return_value = np.sum(self.agent.rewards_list)
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
        for i, action in enumerate(self.agent.actions_list):
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
