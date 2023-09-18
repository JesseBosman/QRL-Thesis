import numpy as np

class ProbabilityAgent():
    def __init__(self, n_holes, print_hole_prob):
        self.n_holes = n_holes
        self.current_probabilities = np.ones(n_holes)/n_holes
        self.print_holeprob = print_hole_prob
        self.action = None
        self.current_policy_sequence = []
        self.longest_policy_sequence = []
    
    def pick_hole(self):
        action = np.argmax(self.current_probabilities)
        self.action = action
        self.current_policy_sequence.append(action)
        max_prob = np.max(self.current_probabilities)
        print("Choose hole {} which had Fox probability {}".format((action+1), max_prob))
        return action
    
    def update_probabilities(self):
        

        
    
