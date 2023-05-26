from minihack_env import * 
from commons import *
import time
import matplotlib.pyplot as plt
from GridWorld import GridWorld
from RandomAgent import RandomAgent
from FixedAgent import FixedAgent
from MCAgent import MCAgent
from SARSAgent import SARSAgent
from QAgent import QAgent
from RLTask import RLTask

# Test the grid world

def task_1_1(n=5, m=5, n_episodes=10000):
    
    print("Hello, this is task 1.1, checking behaviour of random agent.")
    print(f"Running random agent for {n_episodes} episodes . . . ")
    
    # Define the grid world
    env = GridWorld(n, m)
    # Define the agent, action space is N, S, E, W as numbers:
    agent = RandomAgent("0", [0,1,2,3])
    # Define the RL task
    task = RLTask(env, agent)
    
    avg_return_values = task.interact(n_episodes)
    
    print("Done. Plotting average returns . . .")
    plt.figure(figsize=(10,3))
    plt.plot(avg_return_values, '-', color="blue")
    plt.grid()
    plt.title("Random agent")
    plt.xlabel("Episodes")
    plt.ylabel("Average return value")
    plt.savefig("Plots/random_agent/average_return.pdf", bbox_inches = 'tight')
    plt.close()
    
    print("Done. Showing and saving first ten iterations of episode in pixel plots . . .")
    
    # Visualize first ten iterations
    # TODO - how to save them?
    task.visualize_episode(max_number_steps=10, plot=False, name="Plots/random_agent/visualization")
    
def task_1_2(n_episodes=10):
    
    print("Hello, this is task 1.2, checking behaviour of fixed agent.")
    
    title = ""
    name_list = ["Plots/fixed_agent/empty_room/visualization", "Plots/fixed_agent/room_with_lava/visualization"]
    
    for i, id in enumerate([minihack_env.EMPTY_ROOM, minihack_env.ROOM_WITH_LAVA]):
        
        # Get the environment
        print(f"Environment: {id}")
        env = minihack_env.get_minihack_environment(id, add_pixel="True")
        state = env.reset()
        # Define the agent
        agent = FixedAgent("0")
        # Define the RL task
        task = RLTask(env, agent)
        
        avg_return_values = task.interact(n_episodes)
        
        # Visualize first ten iterations
        task.visualize_episode(max_number_steps=10, plot=True, name=name_list[i], title=title)
        
        
##########
# TASK 2 #
##########
        
        
def task_2_MC(n_episodes=100, room_id = minihack_env.EMPTY_ROOM, plot_and_render = True, **kwargs):
    
    print("Hello, this is task 2, MC agent.")
    
    # Get the ID and environment
    id = room_id
    print(f"Environment: {id}")
    save_name = f"Plots/MC_agent/{id}/"
    env = minihack_env.get_minihack_environment(room_id, add_pixel="True")
    state = env.reset()
    # Define the agent
    agent = MCAgent("0", save_name, **kwargs)
    # Define the RL task
    task = RLTask(env, agent, save_name)
    avg_return_values = task.interact(n_episodes)
    
    if plot_and_render:    
        plot_average_returns(avg_return_values, "MC_agent", id, agent.eps)
        # Visualize first ten iterations
        print("Done. Visualizing ten episodes . . .")
        task.visualize_episode(plot=True, name=f"Plots/MC_agent/{id}/visualization")
        
def task_2_SARSA(n_episodes=100, room_id = minihack_env.EMPTY_ROOM, plot_and_render = True, **kwargs):
    
    print("Hello, this is task 2, SARSA agent.")
    
    # Get the ID and environment
    id = room_id
    print(f"Environment: {id}")
    save_name = f"Plots/SARSA_agent/{id}/"
    env = minihack_env.get_minihack_environment(room_id, add_pixel="True")
    state = env.reset()
    # Define the agent
    agent = SARSAgent("0", save_name, **kwargs)
    # Define the RL task
    save_name = kwargs["save_name"] if "save_name" in kwargs else ""
    task = RLTask(env, agent, save_name)
    avg_return_values = task.interact(n_episodes)
    
    if plot_and_render:    
        plot_average_returns(avg_return_values, "SARSA_agent", id, agent.eps)
        
        # Visualize first ten iterations
        print("Done. Visualizing ten episodes . . .")
        task.visualize_episode(plot=True, name=f"Plots/SARSA_agent/{id}/visualization")
       
        
def task_2_Q(n_episodes=100, room_id = minihack_env.EMPTY_ROOM, plot_and_render = True, **kwargs):
    
    print("Hello, this is task 2, Q-learning agent.")
    
    # Get the ID and environment
    id = room_id
    print(f"Environment: {id}")
    save_name = f"Plots/Q_agent/{id}/"
    env = minihack_env.get_minihack_environment(room_id, add_pixel="True")
    state = env.reset()
    # Define the agent
    agent = QAgent("0", save_name, **kwargs)
    # Define the RL task
    task = RLTask(env, agent, save_name)
    avg_return_values = task.interact(n_episodes)
    
    if plot_and_render:    
        print("Done, plotting average returns . . . ")
        plot_average_returns(avg_return_values, "Q_agent", id, agent.eps)
        # Visualize first ten iterations
        print("Done. Visualizing ten episodes . . .")
        task.visualize_episode(plot=True, name=f"Plots/Q_agent/{id}/visualization")
        
def test_saving_and_loading():
    # Learn a lot
    task_2_SARSA(n_episodes=1000)
    # Then load and use the learned policy
    task_2_SARSA(n_episodes=1, load_name = f"Plots/SARSA_agent/empty-room/")
        
def main():
    # task_2_MC(n_episodes=1000)
    # task_2_SARSA(n_episodes=1000)
    # task_2_Q(n_episodes=100)
    test_saving_and_loading()


# Execute main test:
if __name__ == "__main__":
    main()