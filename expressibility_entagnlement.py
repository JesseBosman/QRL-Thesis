from PQC import generate_circuit, generate_model_policy
import cirq
import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
# import sklearn.preprocessing as preprocessing



def compute_fidelities_PQC(set_states):
    fidelities = []

    # iterates over each state wrt to each other state
    for i, state in enumerate(set_states[:-1]):
        fidelities.append([cirq.fidelity(state, state2) for state2 in set_states[(i+1):]])

    return fidelities

def compute_fidelities_HAAR(n_fidelities):
    """"
    Based on relation from: Average fidelity between random quantum states
    by Karol Życzkowski and Hans-Jürgen Sommers
    """
    pass

def plot_pdf_HAAR(n_steps, n_qubits):
    fidelities = np.linspace(0,1, n_steps)
    P = (2**(n_qubits)-1)*(1-fidelities)**(2**(n_qubits)-2)
    plt.xlabel("Fidelity")
    plt.ylabel("Probability")
    plt.plot(fidelities, P)
    plt.show()
    


def create_HAAR_ensemble(n_states,n_qubits):

    states = np.random.uniform(-1, 1, (n_states,n_qubits)) + 1.j * np.random.uniform(-1, 1, (n_states,n_qubits))
    states = []

    for _ in tqdm(range(n_states)):
        state = np.random.uniform(-1, 1, n_qubits) + 1.j * np.random.uniform(-1, 1, n_qubits)
        norm = np.linalg.norm(state)
        state /=norm
        states.append(state)
    
    states = np.array(states)
    with open("HAAR_ensemble_{}.pkl".format(n_states), "wb") as f:
        f.write(states) 

def create_PQC_ensemble(n_states, n_qubits, n_layers):

    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuit, params, inputs = generate_circuit(qubits, n_layers)

    params.extend(inputs)

    state_layer = tfq.layers.State()
    states = []

    for _ in tqdm(range(n_states)):
        param_values = tf.random.uniform(shape = (1,len(params)), minval = -1*np.pi, maxval= np.pi)
        state = state_layer(circuit, symbol_names = params, symbol_values = param_values)
        states.append(state)
    
    states = np.array(states)
    with open("PQC_ensemble_{}states_{}layers.pkl".format(n_states, n_layers), "wb") as f:
        f.write(states) 


create_PQC_ensemble(10000, 5, 5)
# create_HAAR_ensemble(10000,5)

