import gym
import numpy as np
from commons import *

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
                    # the agent:
                    output += '@'
                elif val == 2:
                    # the goal:
                    output += '>'
            output += '\n'
        print(output)
        
        return grid