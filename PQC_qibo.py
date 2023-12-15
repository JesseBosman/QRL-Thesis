import qibo
import tensorflow as tf
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

def one_qubit_rotation(qubit):
    """
    Returns qibo gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [qibo.gates.RX(qubit, 0),
            qibo.gates.RY(qubit, 0),
            qibo.gates.RZ(qubit, 0)]

def one_qubit_rx(qubit):
    """
    Returns qibo gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [qibo.gates.RX(qubit, theta = 0)]

def entangling_layer_CZ(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [qibo.gates.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([qibo.gates.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def entangling_layer_CNOT(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cNOT_ops = [qibo.gates.CNOT(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cNOT_ops += ([qibo.gates.CNOT(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cNOT_ops

def input_layer(qubits, layer, n_inputs):
    lambdas_init = tf.keras.initializers.RandomUniform(minval = -np.pi, maxval= np.pi)
    lambdas = tf.Variable(
            initial_value=lambdas_init(shape=(n_inputs,), dtype="float32"),
            trainable=True, name=f"lambdas{layer}")
    return [qibo.gates.RX(qubit, theta = lambdas[i]) for i, qubit in enumerate(qubits[:n_inputs])]
    


def generate_circuit(qubits, n_layers, n_inputs, RxCnot=False):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)
    
    # Define circuit
    circuit = qibo.Circuit(n_qubits)
    input_parameter_slices = []
    if RxCnot:
        for l in range(n_layers):
            # Variational layer Rx Cnot
            circuit.add(one_qubit_rx(qubit) for qubit in qubits)
            circuit.add(entangling_layer_CNOT(qubits))
            
            # Encoding layer
            
            for i in range(n_inputs):
                circuit.add(one_qubit_rx(qubits[i]))
        
        for i in range(n_inputs):
            indices = []
            for l in range(n_layers):
                
                indices.append(((l+1)*(n_qubits+n_inputs)-n_inputs+i))

            
            input_parameter_slices.append(indices)


                
        # Last varitional layer
        circuit.add(one_qubit_rx(q) for i,q in enumerate(qubits))
    
    else:
        for l in range(n_layers):
            # Variational layer
            circuit.add(one_qubit_rotation(qubit) for qubit in qubits)
            circuit.add(entangling_layer_CZ(qubits))
            # Encoding layer
            for i in range(n_inputs):
                circuit.add(one_qubit_rx(qubits[i]))
                
        for i in range(n_inputs):
            indices = []
            for l in range(n_layers):
                
                indices.append(((l+1)*(3*n_qubits+n_inputs)-n_inputs+i))

            
            input_parameter_slices.append(indices)

        # Last varitional layer
        circuit.add(one_qubit_rotation(qubit) for qubit in qubits)
        

    
    return circuit, np.asarray(input_parameter_slices)

class ObservableWeights(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(ObservableWeights, self).__init__()
        self.w = tf.Variable(
            initial_value= np.random.normal(scale= 0.01, size = output_dim), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        x = tf.multiply(inputs, self.w)
        return x
class ReUploadingPQC_reduced(tf.keras.layers.Layer):
    """
    
    """

    def __init__(self, qubits, n_layers, n_inputs, n_actions, params, name="re-uploading_PQC", RxCnot = False):
        super(ReUploadingPQC_reduced, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.dim_diff = (2**self.n_qubits)-n_actions

        self.circuit, self.input_parameter_slices = generate_circuit(qubits, n_layers, n_inputs, RxCnot=RxCnot)
        self.flattened_indices = self.input_parameter_slices.reshape((n_inputs*n_layers, 1))
        
        self.n_params = len(self.circuit.get_parameters())
        self.thetas = params

    def call(self, inputs):

        input_thetas = tf.gather(self.thetas, indices = self.input_parameter_slices)
        input_thetas = tf.reshape(input_thetas, (self.n_inputs, self.n_layers))
        inputs = tf.reshape(inputs, shape = (2,1))

  
        input_thetas*= inputs

        flattened_thetas = tf.reshape(input_thetas, (self.n_inputs*self.n_layers,))
        params = tf.tensor_scatter_nd_update(tensor = self.thetas, indices= self.flattened_indices, updates =flattened_thetas)
        self.circuit.set_parameters(parameters=params)
        state = self.circuit().state()
        probs = tf.math.real(tf.math.conj(state)*state)
        # print("probs_before")
        # print(probs)
        probs= probs[:-self.dim_diff]
        # print('probs')
        # print(probs)
        sum = tf.reduce_sum(probs)
        # print('sum')
        # print(sum)
        probs /= sum
        # print('sum after')
        # print(tf.reduce_sum(probs))

        
        return  tf.reshape(probs, shape = (1, self.n_actions))
    

    


# @tf.function
def reinforce_update_reduced(states, actions, returns, optimizer, batch_size, eta, params, n_qubits, n_layers, n_actions, RxCnot):
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)
    qubits = np.arange(n_qubits)
    with tf.GradientTape() as tape:
        tape.watch(params)
        model = ReUploadingPQC_reduced(qubits, n_layers, n_inputs=2, n_actions = n_actions, params = params, RxCnot = RxCnot)
        loss = 0
        
        for i, state in enumerate(states):
            state = tf.convert_to_tensor(state)
            logits = model(state)
            entropy_loss = -1*tf.math.reduce_sum(tf.math.multiply(logits, tf.math.log(logits)), axis=1)
            p_actions = logits[0,actions[i,1]]
            log_probs = tf.math.log(p_actions)
            loss += (-log_probs * returns[i] - eta* entropy_loss)/ batch_size
        
    grads = tape.gradient(loss, params)
    optimizer.apply_gradients(zip([grads],[params]))
    return params

