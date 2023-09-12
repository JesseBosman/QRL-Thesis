""""This code closely follows the Tensorflow Quantum tutorial notebook
"Parametrized Quantum Circuits for Reinforcement Learning". Credits therefore are due to the developers of Tensorflow Quantum
"""

import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class Quantum_Circuitry(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, n_actions, activation, beta):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        ops = [cirq.Z(q) for q in self.qubits]
        observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3

        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

        # Assign the model parameters to each optimizer
        self.w_in, self.w_var, self.w_out = 1, 0, 2

        circuit, theta_symbols, input_symbols = self.generate_circuit()

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)       

        self.model=None 
        self.generate_model_policy(beta = beta)



    def one_qubit_rotation(self, qubit, symbols):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        return [cirq.rx(symbols[0])(qubit),
                cirq.ry(symbols[1])(qubit),
                cirq.rz(symbols[2])(qubit)]

    def entangling_layer(self):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(self.qubits, self.qubits[1:])]
        cz_ops += ([cirq.CZ(self.qubits[0], self.qubits[-1])] if len(self.qubits) != 2 else [])
        return cz_ops

    def generate_circuit(self):
        """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
        # Number of qubits
        
        # Sympy symbols for variational angles
        params = sympy.symbols(f'theta(0:{3*(self.n_layers+1)*self.n_qubits})')
        params = np.asarray(params).reshape((self.n_layers + 1, self.n_qubits, 3))
        
        # Sympy symbols for encoding angles
        inputs = sympy.symbols(f'x(0:{self.n_layers})'+f'_(0:{self.n_qubits})')
        inputs = np.asarray(inputs).reshape((self.n_layers, self.n_qubits))
        
        # Define circuit
        circuit = cirq.Circuit()
        for l in range(self.n_layers):
            # Variational layer
            circuit += cirq.Circuit(self.one_qubit_rotation(q, params[l, i]) for i, q in enumerate(self.qubits))
            circuit += self.entangling_layer()
            # Encoding layer
            circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(self.qubits))

        # Last varitional layer
        circuit += cirq.Circuit(self.one_qubit_rotation(q, params[self.n_layers, i]) for i,q in enumerate(self.qubits))
        
        return circuit, list(params.flat), list(inputs.flat)
    
    def ReUpLoadingPQC(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])
    
    
    
    def generate_model_policy(self, beta):
        """Generates a Keras model for a data re-uploading PQC policy."""
        input_tensor = tf.keras.Input(shape=(self.n_qubits, ), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = self.ReUpLoadingPQC([input_tensor])
        process = tf.keras.Sequential([
            Alternating(self.n_actions),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")
        policy = process(re_uploading_pqc)
        self.model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    
    @tf.function
    def reinforce_update(self, states, actions, returns, batch_size):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            logits = self.model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
            
        print('loss')
        print(loss)

        print('trainable variables')
        print(self.model.trainable_variables)


        grads = tape.gradient(loss, self.model.trainable_variables)

        print('grads')
        print(grads)
        for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

