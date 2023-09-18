import tensorflow as tf

class PolicyModel():
    def __init__(self, activation_function='relu', n_hidden_layers=2, n_nodes_per_layer=10, input_dim=4, output_dim=2, learning_rate=0.001):
        self.n_hidden_layers = n_hidden_layers
        self.activation_function = activation_function
        self.n_nodes_per_layer = n_nodes_per_layer
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        self.model = None

        self.build_model()


    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(self.input_dim,)))
        for _ in range(self.n_hidden_layers):
            self.model.add(tf.keras.layers.Dense(self.n_nodes_per_layer, activation=self.activation_function))

        self.model.add(tf.keras.layers.Dense(self.output_dim, activation='softmax'))

    def __call__(self, input):
        return self.model(input)

    def update_reinforce(self, states, actions, returns, batch_size):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)
        

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            logits = self.model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
            # loss = -1*tf.math.multiply(log_probs, returns)

        # print("states")
        
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))