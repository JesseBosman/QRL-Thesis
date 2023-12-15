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


class PickMiddle():
    def __init__(self, n_holes):
        self.current_policy_sequence = []
        self.longest_policy_sequence = []
        self.n_holes = n_holes

    def pick_hole(self):
        return self.n_holes//2

    def update_probabilities(self):
        pass

    def reset(self):
        if len(self.current_policy_sequence) > len(self.longest_policy_sequence):
            self.longest_policy_sequence = self.current_policy_sequence
        
        self.current_policy_sequence = []
        pass


class QuantumProbabilityAgent():
    def __init__(self, n_holes, max_steps=None, prob_1= 0.5, prob_2= 0.5, tunneling_prob=0.2, print_holeprob = False):
        split_1 = (prob_1*(1-tunneling_prob))**(0.5)
        split_2 = (prob_2*(1-tunneling_prob))**(0.5)
        tunnel_split_1 = (prob_1*tunneling_prob)**(0.5)
        tunnel_split_2 = (prob_2*tunneling_prob)**(0.5)
        self.n_holes = n_holes
        if max_steps ==None:
            self.max_steps = 2*(n_holes-2)
        else:
            self.max_steps = max_steps
        self.print_holeprob= print_holeprob
        self.action = None
        self.current_policy_sequence = []
        self.longest_policy_sequence = []
        T_odd = np.zeros(shape= (n_holes, n_holes))
        T_even = np.zeros(shape= (n_holes, n_holes))

        for i in range(n_holes -2):
            T_odd[i,i+1]= split_1
            T_odd[i+2, i+1]= split_2
            T_even[i, i+1]= split_2
            T_even[i+2, i+1]= -1*split_1

            try:
                T_odd[i,i+2]= tunnel_split_1
            except:
                pass
            try:
                T_odd[i+2, i]= tunnel_split_2
            except:
                pass

            try:
                T_even[i, i+2]= tunnel_split_2
            except:
                pass

            try:
                T_even[i+2, i]= -1*tunnel_split_1
            
            except:
                pass
        
        T_odd[-1,-2] = prob_2**(0.5)
        T_odd[0,1] = prob_1**(0.5)

        T_even[-1,-2] = -prob_1**(0.5)
        T_even[0,1] = prob_2**(0.5)
      
        # Hier nog tunneling aanpassen!

        T_odd[1,0]=(1-tunneling_prob)
        T_odd[2,0]= ((1-tunneling_prob)*tunneling_prob)**(0.5)
        T_odd[-2,-1]= (1-tunneling_prob)
        T_odd[-3,-1]= ((1-tunneling_prob)*tunneling_prob)**(0.5)
        T_even[1,0]=-(1-tunneling_prob)
        T_even[2,0]= -((1-tunneling_prob)*tunneling_prob)**(0.5)
        T_even[-2,-1]= (1-tunneling_prob)
        T_even[-3,-1]= ((1-tunneling_prob)*tunneling_prob)**(0.5)

        T_odd[-1,0]= tunneling_prob**(0.5)
        T_odd[0,-1]= tunneling_prob**(0.5)

        T_even[-1,0]= tunneling_prob**(0.5)
        T_even[0,-1]= -tunneling_prob**(0.5)

        self.T_odd = T_odd
        self.T_even = T_even
        
        self.possible_fox_states= np.eye(n_holes)
        self.move_counter = 0
    
    def pick_hole(self):
        
        if self.current_policy_sequence == []:
            action = 1
        else:
            hole_probs = np.sum(self.possible_fox_states**2, axis=1)
            action = np.argmax(hole_probs)
        self.action = action
        self.current_policy_sequence.append(action)
        if self.print_holeprob:
            print("The probabilities per hole was {} so the hole choosen was {}".format(np.sum(self.possible_fox_states**2, axis = 1), (self.action)))
        return action
    
    def update_probabilities(self):
        possible_fox_states = self.possible_fox_states
        possible_fox_states[self.action, :]=0
        if self.move_counter%2==0:
            possible_fox_states = np.matmul(self.T_even, possible_fox_states)
        else:
            possible_fox_states = np.matmul(self.T_odd, possible_fox_states)

        norms = np.linalg.norm(possible_fox_states, axis = 0)
        norms = np.where(norms!=0, norms, 1)
        self.possible_fox_states = possible_fox_states/norms
        self.move_counter+=1
        pass

    def reset(self):
        if len(self.current_policy_sequence) > len(self.longest_policy_sequence):
            self.longest_policy_sequence = self.current_policy_sequence
        
        self.current_policy_sequence = []
        self.possible_fox_states = np.eye(self.n_holes)
        self.move_counter = 0