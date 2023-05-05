from minihack_env import * 
# from quickstart import *
from commons import *
import matplotlib.pyplot as plt

# Test the grid world


def task_1_1(n=5, m=5, n_episodes=10000):
    
    print("Hello, this is task 1.1, checking behaviour of random agent.")
    print(f"Running random agent for {n_episodes} episodes . . . ")
    
    # Define the grid world
    env = GridWorld(n, m)
    # Define the agent
    agent = RandomAgent("0", ACTIONS)
    # Define the RL task
    task = RLTask(env, agent)
    
    avg_return_values = task.interact(n_episodes)
    
    plt.plot(avg_return_values, '-o', color="blue")
    plt.grid()
    plt.title("Random agent")
    plt.xlabel("Episodes")
    plt.ylabel("Average return value")
    plt.savefig("results.pdf", bbox_inches = 'tight')
    plt.close()
    
    # Visualize first ten iterations
    task.visualize_episode(max_number_steps=10)
    

def main():
    task_1_1()


# Execute main test:
if __name__ == "main":
    main()