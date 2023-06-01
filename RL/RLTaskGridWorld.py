import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm
import gym 
from commons import *
import pickle
from GridWorld import GridWorld

class RLTaskGridWorld():

    def __init__(self, env, agent, save_name, max_steps_episode):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env   = env
        self.agent = agent
        self.save_name = save_name + "returns.csv"
        self.actions_list = []
        self.rewards_list = []
        self.max_steps_episode = max_steps_episode
        

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
            state = self.env.reset()
            return_value = 0
            done = False
            reward = 0
            
            self.actions_list = []
            self.rewards_list = []
            
            # Do an episode:
            iteration_counter = 0
            while not done:
                # Agent chooses interaction and interacts with environment
                action = self.agent.act(state, reward)  
                state, reward, done, _ = self.env.step(action)
                # Save chosen action and observed reward
                self.actions_list.append(action)
                self.rewards_list.append(reward)
                # Iteration ends, make agent learn (if learning agent)
                iteration_counter += 1
                
                # Stop after max steps episode is reached
                if iteration_counter >= self.max_steps_episode:
                    break
            
            # Episode is over, compute return
            return_value = np.sum(self.rewards_list)
            # Append the return to the list of returns
            returns_list[i] = return_value
            # Compute the average of the first episodes
            avg_returns_list[i] = np.mean(returns_list[:i+1])
            write_to_txt(self.save_name, [return_value])
            
        return avg_returns_list

        
        
    def visualize_episode(self, agent_id, max_number_steps = 10):
        """
        My own visualization for task 1.1, visualizing final episode
        """
        
        my_cmap = ListedColormap(["white", "black", "green"])
        agent_name = agent_id.replace("_", " ")
        self.env.reset()
        matrix = self.env.render()
        
        i = 0
        reward = 0
        done=False
        action="Start"
        
        fig, axs = plt.subplots(2, 5)
        
        for i in range(max_number_steps):
            ax = axs[i//5, i%5]
            if done:
                break
            # Plot previous state
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Plot the data using the custom colormap
            ax.imshow(matrix, cmap=my_cmap)
            if i == 0:
                ax.set_title(f"Start", fontsize = 12)
            else: 
                ax.set_title(f"t = {i+1}, {action}", fontsize = 12)
            # Take next step
            action = self.actions_list[i]
            # Otherwise, can make a step in the environment
            state, _, done, _ = self.env.step(action)
            # Translate this action into readable language
            action = minihack_env.translate_action(action)
            # Render the environment & get output as matrix:
            matrix = self.env.render()
            if done:
                break
            i += 1
       
        plt.savefig(f"Plots/{agent_id}/visualization.png", bbox_inches='tight')
        plt.close()
                

        
