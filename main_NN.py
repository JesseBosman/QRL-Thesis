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
save_data = True
print_model_summary = True
print_policy = True

env_name = "FoxInAHolev2"
len_state = 2
exp_key = f"{len_state}-inp-enc-rewardfound1"
n_episodes = 1000000
n_holes = 10
batch_size= 10
n_actions = n_holes
state_bounds = 1
gamma = 1
input_dim = len_state
learning_rate = 0.0001
averaging_window = 5000
n_hidden_layers=3
n_nodes_per_layer=100
activation = 'elu'
anil= 0.25
start = 1

save_length = True
save_reward = True

n_reps = 10

agent = reinforce_agent(batch_size=batch_size)

# Start training the agent

for _ in range(n_reps):
    model = PolicyModel(activation_function=activation, n_hidden_layers= n_hidden_layers, n_nodes_per_layer=n_nodes_per_layer, input_dim= input_dim, output_dim=n_actions, learning_rate= learning_rate)
    episode_reward_history = []
    episode_length_history = []
    
    for batch in tqdm(range(n_episodes // batch_size)):
        # Gather episodes
        
        episodes = agent.gather_episodes(state_bounds, n_holes, n_actions, model, batch_size, env_name, len_state=len_state)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([agent.compute_returns(ep_rwds, gamma) for ep_rwds in rewards])

        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])
        
        # Update model parameters.
        eta = max(start*(1-(batch*batch_size)/(n_episodes*anil)), 0)
        model.update_reinforce(states, id_action_pairs, returns, batch_size=batch_size, eta = eta)
        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))
            episode_length_history.append(len(ep_rwds))

        
        if print_avg:
            avg_rewards = np.mean(episode_reward_history[-averaging_window:])
            print('Finished episode', (batch + 1) * batch_size,
            'Average rewards: ', avg_rewards)

    if plotting:
        plot(episode_reward_history, "NN", averaging_window)

    if save_data:

        # the path to where we save the results. we take the first letter of every _ argument block to determine this path
        # directory = f"/data1/bosman/resultsQRL/NN/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}/'
        if save_reward:
            # WORKSTATION
            directory = f"/home/s2025396/data1/resultsQRL/NN/ep_reward/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}"+f"{activation}/"
            # # ALICE
            # directory = f"/home/s2025396/data1/ResultsQRL/NN/"+exp_key+f"{n_holes}holes"+f"{n_hidden_layers}layers"+f"{n_nodes_per_layer}nodes"+f"lr{learning_rate}"+f"neps{n_episodes}"+f"bsize{batch_size}"+f"gamma{gamma}"+f"start{start}"+f"anil{anil}"+f"{activation}/"
            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_reward_history)
        
        if save_length:
            # WORKSTATION
            directory = f"/home/s2025396/data1/resultsQRL/NN/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}"+f"{activation}/"
            # # ALICE
            # directory = f"/home/s2025396/data1/ResultsQRL/NN/"+exp_key+f"{n_holes}holes"+f"{n_hidden_layers}layers"+f"{n_nodes_per_layer}nodes"+f"lr{learning_rate}"+f"neps{n_episodes}"+f"bsize{batch_size}"+f"gamma{gamma}"+f"start{start}"+f"anil{anil}"+f"{activation}/"
            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_length_history)

    if print_policy:
        state = tf.convert_to_tensor([-1*np.ones(len_state)])
        policy = []
        for _ in range(2*(n_holes-2)):
            action = np.random.choice(n_holes, p = model(state).numpy()[0])
            policy.append(action)
            ar_state = state.numpy()
            ar_state[0] = np.roll(ar_state[0], 1)
            ar_state[0][0] = action
            state = tf.convert_to_tensor(ar_state)

        print("Final policy is the following sequence: {}".format(policy))


if print_model_summary:
    model.model.summary()
