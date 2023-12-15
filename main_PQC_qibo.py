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
plot_distribution = False
save_length = True
save_reward = True
RxCnot = True

env_name = "QFIAHv2"
len_state = 2
prob_1= 3/14
prob_2= 11/14
n_episodes = 250000
n_holes = 5
max_steps = 2*(n_holes-2)
n_layers = 1
n_qubits = 3
batch_size = 1
n_actions = n_holes
state_bounds = 1
gamma = 1
input_dim = len_state
averaging_window = 5000



if RxCnot:
    if env_name.lower() == 'qfiahv1':
        exp_key = f"{len_state}-inp-{env_name}-prob1{round(prob_1,2)}-prob2{round(prob_2,2)}-maxsteps{max_steps}-red_PQC-RxCNOT"
    
    else:
        exp_key = f"{len_state}-inp-{env_name}-maxsteps{max_steps}-red_PQC-RxCNOT"

else:
    if env_name.lower() == 'qfiahv1':
        exp_key = f"{len_state}-inp-{env_name}-prob1{round(prob_1,2)}-prob2{round(prob_2,2)}-maxsteps{max_steps}-red_PQC"
    
    else:
        exp_key = f"{len_state}-inp-{env_name}-maxsteps{max_steps}-red_PQC"

anil= 0.25
start = 1

lr = 0.01

n_reps = 10

print("Hyperparameters are:")
print("lr: {}".format(lr))
print("N layers: {}".format(n_layers))
print("N holes: {}".format(n_holes))
print("nqubits {}".format(n_qubits))
print("Len state: {}".format(len_state))
print("RxCnot is {}".format(RxCnot))

# Start training the agent
# for _ in range(n_reps):
def run():
    from REINFORCE import reinforce_agent
    # from PQC import generate_model_policy, reinforce_update
    from PQC_qibo import ReUploadingPQC_reduced, reinforce_update_reduced
    import tensorflow as tf
    agent = reinforce_agent(batch_size=batch_size)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    episode_reward_history = []
    episode_length_history = []
    # As the different sets of parameters require different learning rates, create seperate optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
    w_init = tf.random_uniform_initializer(minval=-np.pi, maxval=np.pi)
    if RxCnot:
        params = tf.Variable(
                initial_value= w_init(shape=((n_qubits+len_state)*n_layers+n_qubits,)),
                trainable=True,
                name = "ReUploadingPQCweights"
            )
    else:

        params = tf.Variable(
                initial_value= w_init(shape=((3*n_qubits+len_state)*n_layers+3*n_qubits,)),
                trainable=True,
                name = "ReUploadingPQCweights"
            )
    
    n_batches = n_episodes//batch_size
    
    for batch in tqdm(range(n_batches)):
        with tf.GradientTape() as tape:
            # # Gather episodes

            model = ReUploadingPQC_reduced(qubits = np.arange(n_qubits), n_layers=n_layers,
                                        n_inputs = len_state, n_actions = n_actions, params= params, RxCnot= RxCnot)
            episodes = agent.gather_episodes(state_bounds, n_holes, n_actions, model, batch_size, env_name, len_state, max_steps= max_steps, prob_1=prob_1, prob_2=prob_2)

            # Group states, actions and returns in numpy arrays
            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([agent.compute_returns(ep_rwds, gamma) for ep_rwds in rewards])

            returns = np.array(returns, dtype=np.float32)

            eta = max(start*(1-(batch*batch_size)/(n_episodes*anil)), 0)
            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            actions = tf.convert_to_tensor(id_action_pairs)
            returns = tf.convert_to_tensor(returns)
    
        
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
            
            # Update model parameters.
            # params = reinforce_update_reduced(states, id_action_pairs, returns, optimizer, batch_size, eta, params, n_qubits, n_layers, n_actions, RxCnot)

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
            directory = f"/home/s2025396/data1/resultsQRL/PQC/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lr{lr}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"
            
            # # WORKSTATION
            # directory = f"/data1/bosman/resultsQRL/PQC/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lr{lr}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"

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
            directory = f"/home/s2025396/data1/resultsQRL/PQC/ep_reward/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lr{lr}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"

            # # WORKSTATION
            # directory = f"/data1/bosman/resultsQRL/PQC/ep_reward/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lr{lr}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"
            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_reward_history)

    if print_model_summary:
        print(model.circuit.summary())



try:
    n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']
    print("cores provided are: "+f"{n_cores}")
    n_cores = int(n_cores)

except:
    
    n_cores = n_reps

print("The number of cores available is {}".format(n_cores))

def test_run():
    for i in tqdm(range(1000)):
        x = i*i

if __name__ == '__main__':     
    # run()
    p = mp.Pool(int(n_cores))
    res = p.starmap(run, [() for _ in range(n_reps)])
    p.close()
    p.join()
    