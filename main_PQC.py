from REINFORCE import reinforce_agent
from PQC import generate_model_policy, reinforce_update
import tensorflow as tf
import numpy as np
from plot import plot
from tqdm import tqdm
import datetime
import os

# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = True

env_name = "FoxInAHole"
n_episodes = 100000
n_holes = 5
n_layers = 5
batch_size=1000
n_actions = n_holes
state_bounds = 1
gamma = 1
input_dim = 2*n_holes -2
averaging_window = 5000

lr_in= 0.1
lr_var= 0.01
lr_out=0.1

n_reps = 5

agent = reinforce_agent(batch_size=batch_size)

# As the different sets of parameters require different learning rates, create seperate optimizers
optimizer_in = tf.keras.optimizers.Adam(learning_rate=lr_in, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=lr_var, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=lr_out, amsgrad=True)

# Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2

optimizers = [optimizer_in, optimizer_var, optimizer_out]
ws= [w_in, w_var, w_out]



# Start training the agent
for _ in range(n_reps):
    episode_reward_history = []
    model = generate_model_policy(n_qubits= input_dim, n_layers= n_layers, n_actions= n_actions, beta= 1)
    for batch in tqdm(range(n_episodes // batch_size)):
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

        
        if print_avg:
            avg_rewards = np.mean(episode_reward_history[-averaging_window:])
            print('Finished episode', (batch + 1) * batch_size,
            'Average rewards: ', avg_rewards)
            
    if plotting:
        plot(episode_reward_history, "NN", averaging_window)

    if save_data:

        # the path to where we save the results. we take the first letter of every _ argument block to determine this path
        directory = f"/data1/bosman/resultsQRL/PQC/"+f'{n_holes}holes'+'layers'+f'{n_layers}'+'nodes'+f'lrin{lr_in}'+'lr'+f'lrvar{lr_var}'+f'lrout{lr_out}'+f'n_eps{n_episodes}/'
            
        if not os.path.isdir(directory):
            os.mkdir(directory)

        print(f"Storing results in {directory}")

        # add date and time to filename to create seperate files with the same setting.
        dt = str(datetime.datetime.now()).split()
        time = f";".join(dt[1].split(":"))

        directory += f"-".join((dt[0], time)) + ".npy"


        np.save(directory, episode_reward_history)
