from REINFORCE import reinforce_agent
from NN import PolicyModel
import numpy as np
from plot import plot
from tqdm import tqdm
import os
import datetime
import tensorflow as tf

# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = False
print_model_summary = False
print_policy = True

env_name = "FoxInAHole"
exp_key = "2(n-2)-inp-enc"
n_episodes = 100000
n_holes = 5
batch_size=100
n_actions = n_holes
state_bounds = 1
gamma = 1
input_dim = 2*(n_holes -2)
learning_rate = 0.005
averaging_window = 5000
n_hidden_layers=2
n_nodes_per_layer=10

n_reps = 1

agent = reinforce_agent(batch_size=batch_size)

# Start training the agent

for _ in range(n_reps):
    model = PolicyModel(n_hidden_layers= n_hidden_layers, n_nodes_per_layer=n_nodes_per_layer, input_dim= input_dim, output_dim=n_actions, learning_rate= learning_rate)
    episode_reward_history = []
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
        model.update_reinforce(states, id_action_pairs, returns, batch_size=batch_size)

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
        # directory = f"/data1/bosman/resultsQRL/NN/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}/'
        
        
        directory = f"/home/s2025396/data1/ResultsQRL/NN/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}/'
        if not os.path.isdir(directory):
            os.mkdir(directory)

        print(f"Storing results in {directory}")

        # add date and time to filename to create seperate files with the same setting.
        dt = str(datetime.datetime.now()).split()
        time = f";".join(dt[1].split(":"))

        directory += f"-".join((dt[0], time)) + ".npy"


        np.save(directory, episode_reward_history)

if print_policy:
    state = tf.convert_to_tensor([-1*np.ones(input_dim)])
    print(state)
    for step in range(input_dim):
        print(np.shape(model(state)))
        action = np.argmax(model(state)[0])
        state[0][step] = action
    
    print("Final policy is the following sequence: {}".format(state))


if print_model_summary:
    model.model.summary()
