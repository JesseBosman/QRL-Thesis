import gym
import numpy as np

class FoxInAHole(gym.Env):
    def __init__(self, n_holes=5, env_config={}):
        self.n_holes = n_holes
        self.hole_nr = None
        self.state = None
        self.guess_counter = 0
        #super.__init__()

    def reset(self):
        # reset the environment to initial random state
        self.hole_nr = np.random.randint(low=0,high=self.n_holes, size= 1)
        self.state = -1*np.ones(int(2*(self.n_holes-2))) #2(n-2) encoding
        # self.state = -1*np.ones(int(self.n_holes)) 
        self.guess_counter = 0
        return self.state
    
    def move_fox(self):
        max_hole_nr = self.n_holes-1
        hole_nr = self.hole_nr
        if hole_nr < max_hole_nr and hole_nr > 0:
            coin = np.random.random(1)

            if coin < 0.5:
                self.hole_nr -= 1
            else:
                self.hole_nr += 1
            
        elif hole_nr == max_hole_nr:
            self.hole_nr -= 1
        
        else:
            self.hole_nr += 1
        

    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            observation (np.array): numpy array with the observation according to the encoding (0 for no fox, 1 for maybe fox, 2 for fox)
            reward (int): the reward for the guess

        """

        if action == self.hole_nr:
            reward = 1
            self.state[self.guess_counter]=action
            done = True
        
        elif self.guess_counter == 2*self.n_holes-5:
            reward = -1
            self.state[self.guess_counter]=action
            done = True
        else:
            reward = -1
            self.state[self.guess_counter]=action
            done = False

            self.move_fox()
        
        self.guess_counter+=1


        return self.state, reward, done, {}
    
    def unbounded_step(self, action):
        if action == self.hole_nr:
            reward = 1
            done = True
        
        else:
            reward = -1
            done = False
            self.move_fox()
        
        return self.state, reward, done, {}
            
class FoxInAHolev2(gym.Env):
    def __init__(self, n_holes=5, env_config={}, len_state = 4):
        self.n_holes = n_holes
        self.hole_nr = None
        self.state = None
        self.guess_counter = 0
        self.len_state = len_state
        #super.__init__()

    def reset(self):
        # reset the environment to initial random state
        self.hole_nr = np.random.randint(low=0,high=self.n_holes, size= 1)
        self.state = -1*np.ones(int(self.len_state)) #n previous picks encoding
        # self.state = -1*np.ones(int(self.n_holes)) 
        self.guess_counter = 0
        return self.state
    
    def move_fox(self):
        max_hole_nr = self.n_holes-1
        hole_nr = self.hole_nr
        if hole_nr < max_hole_nr and hole_nr > 0:
            coin = np.random.random(1)

            if coin < 0.5:
                self.hole_nr -= 1
            else:
                self.hole_nr += 1
            
        elif hole_nr == max_hole_nr:
            self.hole_nr -= 1
        
        else:
            self.hole_nr += 1
        

    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            observation (np.array): numpy array with the observation according to the encoding (0 for no fox, 1 for maybe fox, 2 for fox)
            reward (int): the reward for the guess

        """

        if action == self.hole_nr:
            reward = 1
            state = np.roll(self.state, 1)
            state[0]=action
            self.state = state
            done = True
        
        elif self.guess_counter == 2*self.n_holes-5:
            reward = -1
            state = np.roll(self.state, 1)
            state[0]=action
            self.state = state
            done = True
        else:
            reward = -1
            state = np.roll(self.state, 1)
            state[0]=action
            self.state = state
            done = False

            self.move_fox()
        
        self.guess_counter+=1


        return self.state, reward, done, {}
    
    def unbounded_step(self, action):
        if action == self.hole_nr:
            reward = 1
            done = True
        
        else:
            reward = -1
            done = False
            self.move_fox()
        
        return self.state, reward, done, {}
    
class FoxInAHoleBounded(gym.Env):
    def __init__(self, n_holes=5, env_config={}, len_state = None):
        self.n_holes = n_holes
        self.hole_nr = None
        self.state = None
        self.guess_counter = 0
        #super.__init__()

    def reset(self):
        # reset the environment to initial random state
        self.hole_nr = np.random.randint(low=0,high=self.n_holes, size= 1)
        self.state = -1*np.ones(int(2*(self.n_holes-2))) #2(n-2) encoding
        # self.state = -1*np.ones(int(self.n_holes)) 
        self.guess_counter = 0
        return self.state
    
    def move_fox(self):
        max_hole_nr = self.n_holes-1
        hole_nr = self.hole_nr
        if hole_nr < max_hole_nr and hole_nr > 0:
            coin = np.random.random(1)

            if coin < 0.5:
                self.hole_nr -= 1
            else:
                self.hole_nr += 1
            
        elif hole_nr == max_hole_nr:
            self.hole_nr -= 1
        
        else:
            self.hole_nr += 1
        

    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            observation (np.array): numpy array with the observation according to the encoding (0 for no fox, 1 for maybe fox, 2 for fox)
            reward (int): the reward for the guess

        """

        if action == self.hole_nr:
            reward = 1
            self.state[self.guess_counter]=action
            done = True
        
        elif self.guess_counter == 2*self.n_holes-5:
            reward = -1
            self.state[self.guess_counter]=action
            done = True
        else:
            reward = 0
            self.state[self.guess_counter]=action
            done = False

            self.move_fox()
        
        self.guess_counter+=1


        return self.state, reward, done, {}
    
    def unbounded_step(self, action):
        if action == self.hole_nr:
            reward = 1
            done = True
        
        else:
            reward = -1
            done = False
            self.move_fox()
        
        return self.state, reward, done, {}