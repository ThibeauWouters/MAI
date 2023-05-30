import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gym 
from commons import *
import pickle

class RLTask():

    def __init__(self, env: gym.Env, agent: AbstractAgent, save_name):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env   = env
        self.agent = agent
        # For saving the returns during a run:
        self.save_name = save_name + "returns.csv"
        # Only save if agent has not loaded memory
        
        if len(self.agent.load_name) == 0:
            print("RLTask is saving returns")
            self.save_returns = True
            f = open(self.save_name, "w")
            f.close()
        else:
            print("RLTask is not saving returns")
            self.save_returns = False

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
            # Convert state to the cropped representation
            state = get_crop_chars_from_observation(observation)
            return_value = 0
            done = False
            reward = 0
            
            # Clear states, actions and rewards lists
            self.agent.reset_lists()
            
            # Do an episode:
            iteration_counter = 0
            while not done:
                # Append latest state
                self.agent.states_list[iteration_counter] = np_hash(state)
                # Agent chooses interaction and interacts with environment
                action = self.agent.act(state, reward)  
                observation, reward, done, _ = self.env.step(action)
                next_state = get_crop_chars_from_observation(observation)
                # Save chosen action and observed reward
                self.agent.actions_list[iteration_counter] = action
                self.agent.rewards_list[iteration_counter] = reward
                # Iteration ends, make agent learn
                self.agent.onIterationEnd(iteration_counter, np_hash(next_state))
                # Update state and counter
                state = next_state
                iteration_counter += 1
            
            # Let the agent learn after end of episode:
            self.agent.onEpisodeEnd(iteration_counter)
            
            # Episode is over, compute return
            return_value = np.sum(self.agent.rewards_list)
            # Append the return to the list of returns
            returns_list[i] = return_value
            # Compute the average of the first episodes
            avg_returns_list[i] = np.mean(returns_list[:i+1])
            
            if self.save_returns:
                # If filename given, save avg return to filename
                write_to_txt(self.save_name, [return_value])
                # Save agent's policy for observation later on:
                self.agent.save_memory()
            
        return avg_returns_list

    def visualize_episode(self, agent_id, max_number_steps: int = 25, plot=True,
                          plot_Q = False, name="visualization", title=""):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """
        
        if plot_Q:
            nb_plots = 2
        else:
            nb_plots = 1
        
        # Reset environment
        fsize = 16  # font size for the title
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
        done=False
        for i, action in enumerate(self.agent.actions_list):
            # Check if done, then finished
            if done:
                return
            # Otherwise, can make a step in the environment
            state, _, done, _ = self.env.step(action)
            # Translate this action into readable language
            action = minihack_env.translate_action(action)
            # Render the environment to the screen:
            print(f"=== State {i+1}: (action = {action}) === \n")
            self.env.render()
            print("\n")
            # Save pixel plots to .png if desired
            if plot and not done:
                # Plot the state
                plt.subplot(1, nb_plots, 1)
                plt.imshow(get_crop_pixel_from_observation(state))
                plt.xticks([])
                plt.yticks([])
                plt.title(f"t = {i+1} ({agent_id}, {action})", fontsize = fsize)
                
                # Plot the Q values for that state as well
                if plot_Q:
                    plt.subplot(1, nb_plots, 2)
                    # Get Q values, need char not pixel state
                    char_state = get_crop_chars_from_observation(state)
                    q_vals = [self.agent.Q[(np_hash(char_state), a)] for a in [0, 1, 2, 3]]
                    plt.bar(["N", "E", "S", "W"], q_vals, color="blue", zorder=100)
                    plt.grid()
                    plt.title("Q values")
                # Save and close
                plt.savefig(f"{name}_{i+1}.png", bbox_inches='tight')
                plt.close()
            # Stop rendering after specified max number steps is reached
            if i+1 == max_number_steps:
                return

    def mozaic_episode(self, agent_id, name):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """
        
        # Reset environment
        fsize = 12  # font size for the title
        done = False
        state = self.env.reset()
        
        fig, axs = plt.subplots(2, 5, figsize = (8,4))
        
        i = 0
        reward = 0
        for row in axs:
            for ax in row:
                if done:
                    break
                # Plot previous state
                ax.imshow(get_crop_pixel_from_observation(state))
                # Make a step in the environment
                action = self.agent.actions_list[i]
                # Otherwise, can make a step in the environment
                state, _, done, _ = self.env.step(action)
                # Translate this action into readable language
                action = minihack_env.translate_action(action)
                # Render the environment to the screen:
                self.env.render()
                if done:
                    break
                if i == 0:
                    ax.set_title(f"Start", fontsize = fsize)
                else: 
                    ax.set_title(f"t = {i+1}, {action}", fontsize = fsize)
                    
                i += 1
                
        # Now remove all ticks:
        for row in axs:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
                
        # Save it
        plt.savefig(f"{name}mozaic.png", bbox_inches='tight')
        plt.close()
        
