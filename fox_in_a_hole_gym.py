import gym
import numpy as np

class FoxInAHole(gym.Env):
    def __init__(self, n_holes=5, env_config={}):
        self.n_holes = n_holes
        self.state = None

    def reset(self):
        # reset the environment to initial random state
        hole_nr = np.random.randint(0,self.n_holes,1)
        self.state = hole_nr
        observation = np.ones(self.n_holes)
        return observation
    
    def move_fox(self):
        hole_nr = self.state
        max_hole_nr = self.n_holes-1

        if hole_nr < max_hole_nr and hole_nr > 0:
            coin = np.random.random(1)

            if coin < 0.5:
                self.state -= 1
            else:
                self.state += 1
            
        elif hole_nr == max_hole_nr:
            self.state -= 1
        
        else:
            self.state += 1
        

    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            observation (np.array): numpy array with the observation according to the encoding (0 for no fox, 1 for maybe fox, 2 for fox)
            reward (int): the reward for the guess

        """
        if action == self.state:
            reward = 1
            observation = np.zeros(self.n_holes)
            observation[action] = 2
            done = True
        else:
            reward = -1
            observation = np.ones(self.n_holes)
            observation[action] = 0
            done = False

            self.move_fox()


        return observation, reward, done, {}