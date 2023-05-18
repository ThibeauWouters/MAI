import minihack_env
from commons import *

class FixedAgent():

    def __init__(self, id):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = minihack_env.ACTIONS
        self.learning = True
        # First, keep on going down, then, keep on going left
        self.first_action = 2
        self.second_action = 1 
        self.fixed_action = self.first_action 
        self.turned = False

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        
        # If we are still following first direction, check if we should turn
        if not self.turned:
            # Else, check if we can still do first action, otherwise turn
            observation = get_crop_chars_from_observation(state)
            position = np.argwhere(observation == 64)[0]
            x, y = position
            n, m = observation.shape
            next_x, next_y = get_next_grid_position(x, y, self.fixed_action, n, m)
            if observation[next_x, next_y] != 46:
                # Turn the agent
                self.fixed_action = self.second_action
                self.turned = True
        
        return self.fixed_action
        

    def onEpisodeEnd(self, reward, episode):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass


