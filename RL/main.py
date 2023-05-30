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

ROOM_IDS = [minihack_env.EMPTY_ROOM, minihack_env.ROOM_WITH_LAVA, minihack_env.CLIFF, minihack_env.ROOM_WITH_MONSTER]

max_episode_steps_dict = {minihack_env.EMPTY_ROOM: 50, minihack_env.ROOM_WITH_LAVA:1000, minihack_env.CLIFF:1000, minihack_env.ROOM_WITH_MONSTER: 1000}

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

        
        
def task_2(agent_name, id, n_episodes=1000, **kwargs):
    
    print(f"Agent: {agent_name}. Environment: {id}")
    
    # Get the details of the environment
    max_episode_steps = kwargs["max_episode_steps"] if "max_episode_steps" in kwargs else max_episode_steps_dict[id]
    kwargs["max_episode_steps"] = max_episode_steps
    
    # Location to save information:
    save_name = f"Plots/{agent_name}/{id}/"
    
    # Set-up agent
    if agent_name == "MC_agent":
        agent = MCAgent("0", save_name, **kwargs)
    elif agent_name == "SARSA_agent":
        agent = SARSAgent("0", save_name, **kwargs)
    elif agent_name == "Q_agent":
        agent = QAgent("0", save_name, **kwargs)
    else:
        print("Agent name not recognized")
        return
        
    # Prepare environment
    env = minihack_env.get_minihack_environment(id, add_pixel="True", max_episode_steps=max_episode_steps)
    state = env.reset()
    print(f"Eps: {agent.eps}")
    print(f"Max steps episode: {agent.max_episode_steps}")
    # Define the RL task
    task = RLTask(env, agent, save_name)
    avg_return_values = task.interact(n_episodes)
    
    print("Done, plotting average returns . . . ")
    plot_average_returns(avg_return_values, f"{agent_name}", id, agent.eps)
    
    # Make visualizations
    
    print("Done. Visualizing episodes . . .")
    task.visualize_episode(agent_name, name=f"Plots/{agent_name}/{id}/plots/visualization")  
    
    print("Done. Visualizing episodes with Q values . . .")
    task.visualize_episode(agent_name, plot_Q=True, name=f"Plots/{agent_name}/{id}/plots/Q")  
    
    # Visualize first ten iterations
    print("Done. Creating mozaic of 10 episodes . . .")
    task.mozaic_episode(agent_name, f"Plots/{agent_name}/{id}/")    
        
        
def main():
    ### Run all agents in all environments
    for agent_name in ["SARSA_agent", "Q_agent"]: # "MC_agent" # MC agent is very slow... 
        for id in [minihack_env.EMPTY_ROOM, minihack_env.ROOM_WITH_LAVA, minihack_env.ROOM_WITH_MONSTER, minihack_env.CLIFF]:
            task_2(agent_name=agent_name, id=id)
    
    # task_2(agent_name="SARSA_agent", id=minihack_env.EMPTY_ROOM, n_episodes=50)
    

# Execute main test:
if __name__ == "__main__":
    main()