import numpy as np
n_holes = 5
n_history = 6
gamma = 1.0

class BruteForce():
    def __init__(self, n_holes, n_history):
        self.probs = np.zeros(shape = [n_holes for _ in range(n_history)])
    
    def run():
        for step in range(n_history):
            for action in range(n_holes):




