from minihack.envs.room import MiniHackRoom
import gym
import minihack
from nle import nethack
from minihack import RewardManager


ACTIONS = tuple(nethack.CompassCardinalDirection)
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



des_monster = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
.....
.....
.....
.....
.....
ENDMAP
REGION: (0,0,20,80), lit, "ordinary"
MONSTER: ('Z', "ghoul"), (2,4)
"""

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



def get_minihack_envirnment(id, **kwargs):

    size = kwargs["size"] if "size" in kwargs else 5
    random = kwargs["random"] if "random" in kwargs else False
    add_pixels = kwargs["add_pixel"] if "add_pixel" in kwargs else False
    max_episode_steps = kwargs["max_episode_steps"] if "max_episode_steps" in kwargs else 1000
    obs = ["chars", ]
    if add_pixels:
        obs = ["chars", "pixel"]

    if id == EMPTY_ROOM:
        env = MiniHackRoom(size = size,
                           max_episode_steps=max_episode_steps,
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
        des_file = des_monster
        if not random:
            cont = """STAIR:(4,4),down \nBRANCH: (0,0,0,0),(1,1,1,1) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs
        )
        env = DoNotResetWhenDead(env, max_episode_steps)
    else:
        raise Exception("Environment %s not found" % str(id))

    return env
