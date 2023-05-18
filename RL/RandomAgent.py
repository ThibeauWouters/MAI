import numpy as np

class RandomAgent():

    def __init__(self, id, action_space):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = action_space
        self.learning = True

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """

        # Choose random action
        return np.random.choice(self.action_space)
        

    def onEpisodeEnd(self):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass
