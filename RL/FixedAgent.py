import minihack_env
from commons import *

class FixedAgent(AbstractAgent):

    def __init__(self, id, action_space=np.array([0,1,2,3], dtype=int), max_episode_steps=50):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        
        super().__init__(id, action_space=action_space, max_episode_steps=max_episode_steps)
        
        # self.action_space = minihack_env.ACTIONS
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
            # observation = get_crop_chars_from_observation(state)
            # print(state)
            position = np.argwhere(state == 64)[0]
            x, y = position
            n, m = state.shape
            next_y, next_x = get_next_grid_position(x, y, self.fixed_action, n, m)
            if state[next_y, next_x] != 46:
                # Turn the agent
                self.fixed_action = self.second_action
                self.turned = True
        
        return self.fixed_action
    
    def onEpisodeEnd(self, iteration_counter):
        # Reset the agent's behaviour
        self.fixed_action = self.first_action
        self.turned=False
        return 
        

    def save_memory(self): 
        pass