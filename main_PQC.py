from REINFORCE import reinforce_agent
from PQC import generate_model_policy, reinforce_update
import tensorflow as tf
import numpy as np

env_name = "foxinahole"
n_episodes = 5000
batch_size=100
n_actions = 5
n_layers =5
state_bounds = 1
gamma = 1
input_dim = 5
averaging_window = 100

agent = reinforce_agent(batch_size=batch_size)

# As the different sets of parameters require different learning rates, create seperate optimizers
optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

# Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2

optimizers = [optimizer_in, optimizer_var, optimizer_out]
ws= [w_in, w_var, w_out]



# Start training the agent
episode_reward_history = []
model = generate_model_policy(n_qubits= input_dim, n_layers= n_layers, n_actions= n_actions, beta= 1)
for batch in range(n_episodes // batch_size):
    # Gather episodes
    
    episodes = agent.gather_episodes(state_bounds, input_dim, n_actions, model, batch_size, env_name)

    # Group states, actions and returns in numpy arrays
    states = np.concatenate([ep['states'] for ep in episodes])
    actions = np.concatenate([ep['actions'] for ep in episodes])
    rewards = [ep['rewards'] for ep in episodes]
    returns = np.concatenate([agent.compute_returns(ep_rwds, gamma) for ep_rwds in rewards])

    returns = np.array(returns, dtype=np.float32)
    id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

    # Update model parameters.
    reinforce_update(states, id_action_pairs, returns, model, ws, optimizers, batch_size=batch_size )

    # Store collected rewards
    for ep_rwds in rewards:
        episode_reward_history.append(np.sum(ep_rwds))

    avg_rewards = np.mean(episode_reward_history[-averaging_window:])

    print('Finished episode', (batch + 1) * batch_size,
          'Average rewards: ', avg_rewards)

    if avg_rewards >= 500.0:
        break