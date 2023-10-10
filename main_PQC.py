import numpy as np
from plot import plot
from tqdm import tqdm
import datetime
import os
import multiprocessing as mp

# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = True
print_model_summary = True
print_policy = True

save_length = True
save_reward = True

env_name = "FoxInAHolev2"
len_state = 5
exp_key = f"{len_state}-inp-PQC-v2"
n_episodes = 250000
n_holes = 5
n_layers = 5
batch_size = 10
n_actions = n_holes
state_bounds = 1
gamma = 1
input_dim = len_state
averaging_window = 5000

anil= 0.25
start = 1

lr_in= 0.01
lr_var= 0.001

n_reps = 10

# Start training the agent
# for _ in range(n_reps):
def run():
    from REINFORCE import reinforce_agent
    from PQC import generate_model_policy, reinforce_update
    import tensorflow as tf
    agent = reinforce_agent(batch_size=batch_size)

    episode_reward_history = []
    episode_length_history = []
    # As the different sets of parameters require different learning rates, create seperate optimizers
    optimizer_in = tf.keras.optimizers.Adam(learning_rate=lr_in, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=lr_var, amsgrad=True)
    
    # Assign the model parameters to each optimizer
    w_in, w_var = 1, 0

    optimizers = [optimizer_in, optimizer_var]
    ws= [w_in, w_var]

    model = generate_model_policy(n_qubits= input_dim, n_layers= n_layers, n_actions= n_actions, beta= 1)
    for batch in tqdm(range(n_episodes // batch_size)):
        # Gather episodes
        
        episodes = agent.gather_episodes(state_bounds, n_holes, n_actions, model, batch_size, env_name, len_state)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([agent.compute_returns(ep_rwds, gamma) for ep_rwds in rewards])

        returns = np.array(returns, dtype=np.float32)

        eta = max(start*(1-(batch*batch_size)/(n_episodes*anil)), 0)
        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model, ws, optimizers, batch_size=batch_size, eta= eta)

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

    if save_data:

        if save_length:

            # the path to where we save the results. we take the first letter of every _ argument block to determine this path

            # ALICE
            directory = f"/home/s2025396/data1/resultsQRL/PQC/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f"lrout{lr_out}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"
            
            # # WORKSTATION
            # directory = f"/data1/bosman/resultsQRL/PQC/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"

            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_length_history)

        if save_reward:

            # the path to where we save the results. we take the first letter of every _ argument block to determine this path
            #ALICE
            directory = f"/home/s2025396/data1/resultsQRL/PQC/ep_reward/"+exp_key+f'{n_holes}holes'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f"lrout{lr_out}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"

            # # WORKSTATION
            # directory = f"/data1/bosman/resultsQRL/PQC/ep_reward/"+exp_key+f'{n_holes}holes'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"
            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_reward_history)

    if print_model_summary:
        model.summary()



try:
    n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']
    n_cores = int(n_cores)

except:
    
    print("cores provided are: "+f"{n_cores}")
    n_cores = n_reps

print("The number of cores available is {}".format(n_cores))

def test_run():
    for i in tqdm(range(1000)):
        x = i*i

if __name__ == '__main__':     
       
    p = mp.Pool(int(n_cores))
    res = p.starmap(run, [() for _ in range(n_reps)])
    p.close()
    p.join()
    