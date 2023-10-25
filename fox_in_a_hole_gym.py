import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq

class FoxInAHole():
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
            
class FoxInAHolev2():
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
    
class FoxInAHoleBounded():
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
    
class QFIAHv1():
    def __init__(self, n_holes, len_state, max_steps=None):
        self.len_state = len_state
        self.n_holes= n_holes
        if max_steps ==None:
            self.max_steps = 2*(n_holes-2)
        else:
            self.max_steps = max_steps
        self.T_odd = np.array([[0, 2**(-0.5), 0, 0, 0],
                               [1, 0, 2**(-0.5), 0, 0],
                               [0, 2**(-0.5), 0, 2**(-0.5), 0],
                               [0, 0, 2**(-0.5), 0, 1],
                               [0, 0, 0, 2**(-0.5), 0]])
        
        self.T_even = np.array([[0, 2**(-0.5), 0, 0, 0],
                               [-1, 0, 2**(-0.5), 0, 0],
                               [0, -2**(-0.5), 0, 2**(-0.5), 0],
                               [0, 0, -2**(-0.5), 0, 1],
                               [0, 0, 0, -2**(-0.5), 0]])

        pass

    def reset(self):

        initial_hole = np.random.randint(0,5,1)
        self.fox_state= np.zeros(self.n_holes)
        self.fox_state[initial_hole]=1
        self.game_state = np.ones(self.len_state)*-1
        self.move_counter = 0


        return self.game_state

    def move_fox(self):
        fox_state =self.fox_state
        if self.move_counter%2==0:
            fox_state= np.matmul(self.T_even, fox_state)
        else:
            fox_state= np.matmul(self.T_odd, fox_state)
        
        norm = np.linalg.norm(fox_state)
        self.fox_state = fox_state/norm
        
        self.move_counter+=1
        
        
    def step(self, action):
        game_state = np.roll(self.game_state, 1)
        game_state[0]=action
        self.game_state = game_state
        prob_correct = self.fox_state[action]**2
        coin = np.random.random(1)

        if prob_correct>coin:
            reward = 1
            done = True
        
        else:
            reward = -1
            if self.move_counter== self.max_steps-1:
                done = True
            else:
                done = False
                fox_state = self.fox_state
                fox_state[action]= 0
                norm = np.linalg.norm(fox_state)
                self.fox_state= fox_state/norm
                self.move_fox()

        return self.game_state, reward, done, {}
        