import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm
import gym 
from commons import *
import pickle
from GridWorld import GridWorld
from MCAgent import MCAgent
import os

class RLTask():

    def __init__(self, env: gym.Env, agent: AbstractAgent, room_id, save_returns=True):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env   = env
        self.agent = agent
        self.room_id = room_id
        # To save the returns:
        self.save_returns = save_returns
        if save_returns:
            print("RLTask is saving returns")
            self.save_name = f"Plots/{agent.id}/{room_id}/returns.csv"
            # Initialize the CSV file for the returns
            self.save_returns = True
            f = open(self.save_name, "w")
            f.close()
        else:
            print("RLTask is NOT saving returns")
            

    def interact(self, n_episodes: int) -> float:
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode average returns (see assignment for a definition)
        """
        
        # Initialize an empty list for the returns
        returns_list = np.zeros(n_episodes)
        avg_returns_list = np.zeros(n_episodes)
        
        # TO save the returns to check if there is a bug
        # Do the episodes
        for i in tqdm(range(n_episodes)):
            # Reset everything
            observation = self.env.reset()
            # Convert state to the cropped representation
            state = get_crop_chars_from_observation(observation)
            return_value = 0
            done = False
            reward = 0
            
            # Clear states, actions and rewards lists
            self.agent.reset_lists()
            
            
            all_rewards_list = []
            
            # Do an episode:
            iteration_counter = 0
            while not done:
                # Append latest state
                self.agent.states_list[iteration_counter] = np_hash(state)
                # Agent chooses interaction and interacts with environment
                action = self.agent.act(state, reward)  
                observation, reward, done, _ = self.env.step(action)
                # Save chosen action and observed reward
                self.agent.actions_list[iteration_counter] = action
                self.agent.rewards_list[iteration_counter] = reward
                # Iteration ends, make agent learn (if learning agent)
                next_state = get_crop_chars_from_observation(observation)
                if self.agent.learning:
                    self.agent.onIterationEnd(iteration_counter, np_hash(next_state))
                # Update state and counter
                state = next_state
                iteration_counter += 1
            
            # Episode is over, compute return
            return_value = np.sum(self.agent.rewards_list)
            
            # Let the agent learn after end of episode:
            if self.agent.learning:
                self.agent.onEpisodeEnd(iteration_counter)
            
            # Append the return to the list of returns
            returns_list[i] = return_value
            # Compute the average of the first episodes
            avg_returns_list[i] = np.mean(returns_list[:i+1])
            
            # If desired, save the average return to filename
            if self.save_returns:
                write_to_txt(self.save_name, [return_value])
            
        return avg_returns_list

    def visualize_episode(self, max_number_steps: int = 25, plot=True, plot_Q = False, render=True):
        
        print(f"Visualizing for max: {max_number_steps}")
        
        # Plotting Q values as well:
        if plot_Q:
            nb_plots = 2
            name = "Q"
            fig, axs = plt.subplots(1, nb_plots, figsize=(11,5), gridspec_kw={'width_ratios': [2, 1]})
        else:
            name = "visualization"
            nb_plots = 1
            
        # Clear directory and specify save location the directory
        directory = f"Plots/{self.agent.id}/{self.room_id}/plots/"
        # for f in os.listdir(directory):
        #     os.remove(f)
        save_location = f"{directory}{name}"
        
        # Reset environment
        fsize = 12  # font size for the title
        state = self.env.reset()
        if render:
            print("=== Starting state: === \n")
            self.env.render()
        if plot:
            plt.imshow(get_crop_pixel_from_observation(state))
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Start", fontsize = fsize)
            plt.savefig(f"{save_location}_{0}.png", bbox_inches='tight')
            plt.close()
        
        # Reconstruct the episode from the actions and information of the environment
        done=False
        agent_name = self.agent.id.replace("_", " ")
        for i, action in enumerate(self.agent.actions_list):
            # Check if done, then finished
            if done:
                return
            # Otherwise, can make a step in the environment
            state, _, done, _ = self.env.step(action)
            # Translate this action into readable language
            action = minihack_env.translate_action(action)
            # Render the environment to the screen:
            if render:
                print(f"=== State {i+1}: (action = {action}) === \n")
                self.env.render()
            # Save pixel plots to .png if desired
            if plot and not done:
                # Plot the state
                plt.subplot(1, nb_plots, 1)
                plt.imshow(get_crop_pixel_from_observation(state))
                plt.xticks([])
                plt.yticks([])
                plt.title(f"t = {i+1} ({agent_name}, {action})", fontsize = fsize)
                
                # Plot the Q values for that state as well
                if plot_Q and i > 0:
                    plt.subplot(1, nb_plots, 2)
                    # Get Q values, need char not pixel state
                    char_state = get_crop_chars_from_observation(state)
                    q_vals = [self.agent.Q[(np_hash(char_state), a)] for a in [0, 1, 2, 3]]
                    if isinstance(self.agent, MCAgent):
                        q_vals = np.array(q_vals)
                        plt.plot([0, 1, 2, 3], q_vals[:, 0], "-o", color="blue", zorder=100)
                    else:
                        plt.plot([0, 1, 2, 3], q_vals, "-o", color="blue", zorder=100)
                    plt.xticks([0, 1, 2, 3], ["N", "E", "S", "W"])
                    plt.grid()
                    plt.title("Q values")
                # Save and close
                plt.savefig(f"{save_location}_{i+1}.png", bbox_inches='tight')
                plt.close()
                
            # Stop rendering after specified max number steps is reached
            if i+1 == max_number_steps:
                return

    def mozaic_episode(self, figsize=(11,5)):
        """
        Create a tile of 10 first steps for the agent, to shown the learned policy in the report.
        """
        
        
        # Reset environment
        fsize = 12  # font size for the title
        done = False
        state = self.env.reset()
        
        fig, axs = plt.subplots(2, 5, figsize=figsize)
        
        # Now remove all ticks:
        reward = 0
        for i in range(10):
            ax = axs[i//5, i%5]
            # Plot previous state
            ax.imshow(get_crop_pixel_from_observation(state))
            if i == 0:
                ax.set_title(f"Start", fontsize = fsize)
            else: 
                ax.set_title(f"t = {i+1}, {action}", fontsize = fsize)
            # Check termination 
            if done:
                for j in range(i, 10):
                    fig.delaxes(axs[j//5, j%5])
                break
            else:
                # Make a step in the environment
                action = self.agent.actions_list[i]
                # Otherwise, can make a step in the environment
                state, _, done, _ = self.env.step(action)
                # Translate this action into readable language
                action = minihack_env.translate_action(action)
            # Increment counter
            i += 1
            
        for row in axs:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
                
        # Save it
        plt.savefig(f"Plots/{self.agent.id}/{self.room_id}/plots/mozaic.png", bbox_inches='tight')
        plt.savefig(f"Plots/{self.agent.id}/{self.room_id}/plots/mozaic.pdf", bbox_inches='tight')
        plt.close()
        
        
