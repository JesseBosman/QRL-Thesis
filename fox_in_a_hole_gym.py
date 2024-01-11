import numpy as np

# import tensorflow as tf


class FIAH:
    """
    Class object for the Fox in hole game. The amount of holes, the max game length and the length of the input state are all adjustable.
    """

    def __init__(self, max_steps=6, len_state=4, transfer_matrices=None):
        self.game_state = None
        self.hole_probs = None
        self.move_counter = 0
        self.len_state = len_state
        self.max_steps = max_steps
        self.transfer_matrix = transfer_matrices

        self.n_holes = np.shape(transfer_matrices)[0]

    def reset(self):
        """
        Resets the environment.
        """
        # reset the environment to initial random state
        initial_hole = np.random.randint(0, self.n_holes, 1)
        self.hole_probs = np.zeros(self.n_holes)
        self.hole_probs[initial_hole] = 1
        self.game_state = np.ones(self.len_state) * -1
        self.move_counter = 0

        return self.game_state

    def move_fox(self):
        """
        Moves the fox randomly to a hole either to the right or to the left.
        """
        hole_probs = self.hole_probs
        hole_probs = np.matmul(self.transfer_matrix, hole_probs)

        norm = np.sum(hole_probs)
        self.hole_probs = hole_probs / norm

        self.move_counter += 1

    def step(self, action):
        """ "
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            state (np.array): numpy array with the last two made guesses (inlcuding the guess made with this functions call).
            reward (int): the reward for the guess.
            done (bool): True if the game is finished, False if not.
            {} (?): Ignore this, its a remnant of the gym environment guidelines.

        """

        prob_correct = self.hole_probs[action]
        coin = np.random.random(1)
        game_state = np.roll(self.game_state, 1)
        game_state[0] = action
        self.game_state = game_state

        if prob_correct > coin:
            reward = 0
            done = True

        else:
            reward = -1
            if self.move_counter == self.max_steps - 1:
                done = True
            else:
                done = False
                self.hole_probs[action] = 0
                self.move_fox()

        return self.game_state, reward, done, {}


class Givens:
    """
    Class object for the Givens wall based Fox in hole game. The amount of holes, the max game length and the length of the input state are all adjustable.
    """

    def __init__(self, max_steps=6, len_state=4, transfer_matrices=None):
        self.game_state = None
        self.fox_state = None
        self.move_counter = 0
        self.len_state = len_state
        self.max_steps = max_steps
        self.transfer_matrix = transfer_matrices

        self.n_holes = np.shape(transfer_matrices)[0]

    def reset(self):
        """
        Resets the environment.
        """
        # reset the environment to initial random state
        initial_hole = np.random.randint(0, self.n_holes, 1)
        self.fox_state = np.zeros(self.n_holes)
        self.fox_state[initial_hole] = 1
        self.game_state = np.ones(self.len_state) * -1
        self.move_counter = 0

        return self.game_state

    def move_fox(self):
        """
        Moves the fox randomly to a hole either to the right or to the left.
        """
        fox_state = self.fox_state
        fox_state = np.matmul(self.transfer_matrix, fox_state)
        norm = np.linalg.norm(fox_state)

        self.fox_state = fox_state/norm
        self.move_counter += 1
        pass

    def step(self, action):
        """
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            state (np.array): numpy array with the last two made guesses (inlcuding the guess made with this functions call).
            reward (int): the reward for the guess.
            done (bool): True if the game is finished, False if not.
            {} (?): Ignore this, its a remnant of the gym environment guidelines.

        """

        prob_correct = np.abs(self.fox_state[action]) ** 2
        coin = np.random.random(1)
        game_state = np.roll(self.game_state, 1)
        game_state[0] = action
        self.game_state = game_state

        if prob_correct > coin:
            reward = 0
            done = True

        else:
            reward = -1
            if self.move_counter == self.max_steps - 1:
                done = True
            else:
                done = False
                self.fox_state[action] = 0
                self.move_fox()

        return self.game_state, reward, done, {}


class QFIAH:
    """
    Class for a quantum version of FoxInAHole, supporting the superposition of the fox across holes and 'tunneling'.
    """

    def __init__(self, max_steps=6, len_state=2, transfer_matrices=None):
        self.game_state = None
        self.hole_probs = None
        self.move_counter = 0
        self.len_state = len_state
        self.max_steps = max_steps
        self.transfer_matrices = transfer_matrices
        self.n_holes = np.shape(self.transfer_matrices[0])[0]

    def reset(self):
        initial_hole = np.random.randint(0, self.n_holes, 1)
        self.hole_probs = np.zeros(self.n_holes)
        self.hole_probs[initial_hole] = 1
        self.game_state = np.ones(self.len_state) * -1
        self.move_counter = 0

        return self.game_state

    def move_fox(self):
        hole_probs = self.hole_probs
        if self.move_counter % 2 == 0:
            hole_probs = np.matmul(self.transfer_matrices[0], hole_probs)
        else:
            hole_probs = np.matmul(self.transfer_matrices[1], hole_probs)

        norm = np.sum(np.abs(hole_probs))
        self.hole_probs = hole_probs / norm

        self.move_counter += 1

    def step(self, action):
        game_state = np.roll(self.game_state, 1)
        game_state[0] = action
        self.game_state = game_state
        prob_correct = self.hole_probs[action]
        coin = np.random.random(1)

        if prob_correct > coin:
            reward = 0
            done = True

        else:
            reward = -1
            if self.move_counter == self.max_steps - 1:
                done = True
            else:
                done = False
                self.hole_probs[action] = 0
                self.move_fox()

        return self.game_state, reward, done, {}
