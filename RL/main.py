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
from RLTaskGridWorld import RLTaskGridWorld

# Test the grid world

ROOM_IDS = [minihack_env.EMPTY_ROOM, minihack_env.ROOM_WITH_LAVA, minihack_env.CLIFF, minihack_env.ROOM_WITH_MONSTER]

max_episode_steps_dict = {minihack_env.EMPTY_ROOM: 50, minihack_env.ROOM_WITH_LAVA:1000, minihack_env.CLIFF:1000, minihack_env.ROOM_WITH_MONSTER: 1000}

def task_1_1(n=5, m=5, n_episodes=10000, max_steps_episode=50, agent_id = "random_agent"):
    
    print("Hello, this is task 1.1, checking behaviour of random agent.")
    print(f"Running random agent for {n_episodes} episodes . . . ")
    
    save_name = f"Plots/{agent_id}/"
    
    # Define the grid world
    env = GridWorld(n, m)
    # Define the agent, action space is N, S, E, W as numbers:
    agent = RandomAgent("0", [0,1,2,3])
    # Define the RL task
    task = RLTaskGridWorld(env, agent, save_name, max_steps_episode)
    
    avg_return_values = task.interact(n_episodes)
    np.savetxt(save_name + "returns.csv", avg_return_values)
    
    task.visualize_episode(agent_id, max_number_steps=10)
    
def task_1_2(n_episodes=1):
    
    print("Hello, this is task 1.2, checking behaviour of fixed agent.")
    
    title = ""
    name_list = ["Plots/fixed_agent/empty-room/visualization", "Plots/fixed_agent/room-with-lava/visualization"]
    env_list = [minihack_env.EMPTY_ROOM, minihack_env.ROOM_WITH_LAVA]
    max_episode_steps_list = [50, 50]
    
    
    for i, id in enumerate(env_list):
        save_name = f"Plots/fixed_agent/{id}/"
        max_episode_steps = max_episode_steps_list[i]
        # Get the environment
        print(f"Environment: {id}")
        # Prepare environment
        env = minihack_env.get_minihack_environment(id, add_pixel="True", max_episode_steps=max_episode_steps)
        state = env.reset()
        # Define the agent
        agent = FixedAgent("0", max_episode_steps=max_episode_steps)
        # Define the RL task
        task = RLTask(env, agent, save_name)
        
        avg_return_values = task.interact(n_episodes)
        
        # Visualize first ten iterations
        task.visualize_episode("fixed agent", max_number_steps=10, plot=True, name=name_list[i], title=title)
        task.mozaic_episode("fixed agent", f"Plots/fixed_agent/{id}/")    
        
        
##########
# TASK 2 #
##########

        
def task_2(agent_name, room_id, n_episodes=1000, **kwargs):
    
    print(f"Agent: {agent_name}. Environment: {room_id}")
    
    # Get the details of the environment
    max_episode_steps = kwargs["max_episode_steps"] if "max_episode_steps" in kwargs else max_episode_steps_dict[room_id]
    kwargs["max_episode_steps"] = max_episode_steps
    
    # Set-up agent
    if agent_name == "MC_agent":
        agent = MCAgent("MC_agent", **kwargs)
    elif agent_name == "SARSA_agent":
        agent = SARSAgent("SARSA_agent", **kwargs)
    elif agent_name == "Q_agent":
        agent = QAgent("Q_agent", **kwargs)
    else:
        print("Agent name not recognized")
        return
        
    # Prepare environment, disable pixel representation to have faster learning
    env = minihack_env.get_minihack_environment(room_id, add_pixel=False, max_episode_steps=max_episode_steps)
    state = env.reset()
    print(f"Eps: {agent.eps}")
    print(f"Max steps episode: {agent.max_episode_steps}")
    # Define the RL task
    task = RLTask(env, agent, room_id, save_returns=False)
    avg_return_values = task.interact(n_episodes)
    # Save return values to txt file
    np.savetxt(f"Plots/{agent.id}/{room_id}/return.txt", avg_return_values)
    
    # Now, get environment again, but now adding the pixel representation for plotting
    env = minihack_env.get_minihack_environment(room_id, add_pixel=True, max_episode_steps=max_episode_steps)
    # Disable exploration (check agent's policy)
    agent.eps = -1
    agent.learning = False
    task = RLTask(env, agent, room_id, save_returns=False)
    state = env.reset()
    # Get a single interaction to check agent behaviour
    _ = task.interact(1)
    
    # print("Done, plotting average returns . . . ")
    # plot_average_returns(avg_return_values, f"{agent_name}", id, agent.eps)
    
    # Make visualizations
    
    print("Done. Visualizing episodes . . .")
    task.visualize_episode(max_number_steps=50)
    
    print("Done. Visualizing episodes with Q values . . .")
    task.visualize_episode(plot_Q=True, render=False)  
    
    # Visualize first ten iterations
    print("Done. Creating mozaic of 10 episodes . . .")
    task.mozaic_episode()   
    
    print("Done") 
        
        
def main():
    
    ### Task 1
    # task_1_1()
    # task_1_2()
    
    ### Task 2
    ## All agents
    
    # for agent_name in ["MC_agent"]: # "SARSA_agent", "Q_agent" # MC agent is very slow... 
    #     for id in [minihack_env.EMPTY_ROOM, minihack_env.ROOM_WITH_LAVA, minihack_env.ROOM_WITH_MONSTER, minihack_env.CLIFF]:
    #         task_2(agent_name=agent_name, id=id)
    
    ## Single agent
    task_2(agent_name="MC_agent", room_id=minihack_env.CLIFF, n_episodes=1000, eps = 0.4, eps_period = 500)
    

# Execute main test:
if __name__ == "__main__":
    main()