from minihack.envs.room import MiniHackRoom
import gym
import minihack
from nle import nethack
from minihack import RewardManager

import numpy as np


ACTIONS = tuple(nethack.CompassCardinalDirection)
print(ACTIONS)
def translate_action(action):
    """Simple auxiliary function that translates an action from ACTION to its compass direction as str."""
    if action == ACTIONS[0]:
        return "N"
    elif action == ACTIONS[1]:
        return "E"
    elif action == ACTIONS[2]:
        return "S"
    elif action == ACTIONS[3]:
        return "W"
    else:
        # If action not recognized, return original
        return action
    
EMPTY_ROOM = "empty-room"
ROOM_WITH_LAVA_MODIFIED = "room-with-lava-modified"
ROOM_WITH_LAVA = "room-with-lava"
ROOM_WITH_MONSTER = "room-with-monster"
CLIFF = "cliff-minihack"


des_room_with_lava ="""
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|-----     ------
|.....-- --.....|
|.T.T...-.....L.|
|...T...........|
|.T.T...-.....L.|
|.....-----.....|
|-----     ------
ENDMAP
"""

des_cliff="""
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|----------------
|...............|
|...............|
|...............|
|...............|
|.LLLLLLLLLLLLL.|
|----------------
ENDMAP
"""

class GridWorld(gym.Env):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(n), gym.spaces.Discrete(m)))
        self.reward_range = (-1, 0)
        self.state = (0, 0)
        self.goal = (n-1, m-1)
        
    def step(self, action):
        # The current state
        x, y = self.state
        # Check which compass direction was chosen
        if action == ACTIONS[0]:
            # Go north
            x = max(0, x-1)
        elif action == ACTIONS[1]:
            # Go east
            y = min(self.m-1, y+1)
        elif action == ACTIONS[2]:
            # Go south
            x = min(self.n-1, x+1)
        elif action == ACTIONS[3]:
            # Go west
            y = max(0, y-1)
            
        # Give rewards after action was performed
        if (x, y) == self.goal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        self.state = (x, y)
        return self.state, reward, done

    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def render(self):
        # Get an empty grid of specified size (n, m)
        grid = np.zeros((self.n, self.m), dtype=np.int8)
        grid[self.goal] = 2
        grid[self.state] = 1

        output = ''
        for row in grid:
            for val in row:
                if val == 0:
                    output += '.'
                elif val == 1:
                    output += 'X'
                elif val == 2:
                    output += 'G'
            output += '\n'
        print(output)


# Methods given by assignment:

"""Reshaping rewards with death levels"""
class DoNotResetWhenDead(gym.Wrapper):
    """Modifies Reward by a constant"""

    def __init__(self, env, max_episode_steps = 1000, goal_reward = 0, negative_step_reward = -1, dead_negative_reward = -100):
        super().__init__(env)
        self.step_counter = 0
        self.max_episode_steps = max_episode_steps
        self.goal_reward = goal_reward
        self.negative_step_reward = negative_step_reward
        self.dead_negative_reward = dead_negative_reward

    def step(self, action):
        # We override the done
        obs, reward, done, info = self.env.step(action)
        self.step_counter += 1
        if info["end_status"] == self.env.StepStatus.ABORTED or self.step_counter == self.max_episode_steps:
            done = True
            reward = self.negative_step_reward
        elif info["end_status"] == self.env.StepStatus.DEATH:
            done = False
            reward = self.dead_negative_reward  + self.negative_step_reward
            obs = self.env.reset()
        elif info["end_status"] == self.env.StepStatus.TASK_SUCCESSFUL:
            done = True
            reward = self.goal_reward

        else:
            done = False
            reward = self.negative_step_reward
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.step_counter = 0
        return obs



def get_minihack_environment(id, **kwargs):

    size = kwargs["size"] if "size" in kwargs else 5
    random = kwargs["random"] if "random" in kwargs else False
    add_pixels = kwargs["add_pixel"] if "add_pixel" in kwargs else False
    max_episode_steps = kwargs["max_episode_steps"] if "max_episode_steps" in kwargs else 1000
    obs = ["chars", ]
    if add_pixels:
        obs = ["chars", "pixel"]

    if id == EMPTY_ROOM:
        env = MiniHackRoom(size = size,
                           max_episode_steps=50,
                           actions=ACTIONS,
                           random = random,
                           observation_keys=obs,
                           reward_win=0,
                           penalty_step = -1, penalty_time=-1)
    elif id == ROOM_WITH_LAVA_MODIFIED:
        des_file = des_room_with_lava
        if not random:
            cont = """STAIR:(15,3),down \nBRANCH: (3,3,3,3),(4,4,4,4) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs
        )

        env = DoNotResetWhenDead(env, max_episode_steps)
    elif id == ROOM_WITH_LAVA:
        des_file = des_room_with_lava
        if not random:
            cont = """STAIR:(15,3),down \nBRANCH: (3,3,3,3),(4,4,4,4) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs,
            penalty_step=-1,
            penalty_time= -1,
            reward_lose = -100,
            reward_win= 0
        )
    elif id == CLIFF:
        des_file = des_cliff
        if not random:
            cont = """STAIR:(15,5),down \nBRANCH: (1,5,1,5),(4,4,4,4) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs
        )
        env = DoNotResetWhenDead(env, max_episode_steps)
    elif id == ROOM_WITH_MONSTER:
        env = MiniHackRoom(size = size,
                           max_episode_steps=max_episode_steps,
                           actions=ACTIONS,
                           random = random,
                           observation_keys=obs,
                           n_monster=1)
        env = DoNotResetWhenDead(env, max_episode_steps)

    else:
        raise Exception("Environment %s not found" % str(id))

    return env
