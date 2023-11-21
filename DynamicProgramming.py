import numpy as np

class QValueIterationAgent():
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_holes, n_history):
        self.n_actions = n_holes
        self.Q_sa = np.zeros(shape = [n_holes for _ in range(n_history+1)])
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s '''
        # TO DO: Add own code
        a = np.argmax(self.Q_sa[s]) # Returns the index of the most advantagous action for state s
       
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        self.Q_sa[s,a] = np.sum((p_sas*(r_sas+self.gamma*np.amax(self.Q_sa, axis = 1))))
        pass
    