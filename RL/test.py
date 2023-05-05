from minihack_env import * 
# from quickstart import *
from commons import *
import matplotlib.pyplot as plt

# Test the grid world


def random_agent_test(n=5, m=5, n_episodes=10):
    
    print("Testing random agent...")
    
    # Define the grid world
    env = GridWorld(n, m)
    # Define the agent
    agent = RandomAgent("0", ACTIONS)
    # Define the RL task
    task = RLTask(env, agent)
    
    return_values = task.interact(n_episodes)
    
    task.visualize_episode()
    

def main():
    random_agent_test()


# Execute main test:
main()