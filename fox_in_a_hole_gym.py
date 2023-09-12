import gym
import numpy as np
from gym.spaces import Discrete
from gym.spaces import MultiBinary

class FoxInAGame(gym.Env):
    def __init__(self,env_config={}):
        self.observation_space = MultiBinary(5)
        
        self.action_space = Discrete(5)

    def reset(self):
        # reset the environment to initial state
        observation = np.array([0,0,0,0,0])
        return observation

    def step(self, action):
        # perform one step in the game logic
        return observation, reward, done, info