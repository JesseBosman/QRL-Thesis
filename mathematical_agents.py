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
        if self.current_policy_sequence == []:
            action = 1
        else:
            action = np.argmax(self.current_probabilities)
        self.action = action
        self.current_policy_sequence.append(action)
        if self.print_holeprob:
            print("The probabilities per hole was {} so the hole choosen was {}".format(self.current_probabilities, (self.action +1)))
        return action
    
    def update_probabilities(self):
        working_probs = self.current_probabilities
        working_probs[self.action] = 0
        working_probs/= np.sum(working_probs)
        working_probs_shiftplus = np.roll(working_probs, 1)
        working_probs_shiftplus[0] = 0
        working_probs_shiftplus[1] *=2
        working_probs_shiftmin = np.roll(working_probs, -1)
        working_probs_shiftmin[-1] = 0
        working_probs_shiftmin[-2] *=2
        self.current_probabilities = 0.5*(working_probs_shiftplus+working_probs_shiftmin)
        pass

    def reset(self):
        if len(self.current_policy_sequence) > len(self.longest_policy_sequence):
            self.longest_policy_sequence = self.current_policy_sequence
        
        self.current_policy_sequence = []
        self.current_probabilities = np.ones(self.n_holes)/self.n_holes
        pass

class BoundAgent():
    def __init__(self, n_holes):
        self.current_policy_sequence = []
        self.longest_policy_sequence = []
        self.previous_action = 0
        self.n_holes = n_holes
        self.direction = 1

    def pick_hole(self):
        if self.previous_action == self.n_holes-2 and self.direction ==1:
            self.direction = -1
            self.current_policy_sequence.append(self.previous_action)
            return self.previous_action
                
        else:
            self.previous_action += self.direction
            self.current_policy_sequence.append(self.previous_action)
            return self.previous_action

    def update_probabilities(self):
        pass

    def reset(self):
        if len(self.current_policy_sequence) > len(self.longest_policy_sequence):
            self.longest_policy_sequence = self.current_policy_sequence
        
        self.current_policy_sequence = []
        self.previous_action = 0
        self.direction = 1
        pass


