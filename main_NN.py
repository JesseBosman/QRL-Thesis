import numpy as np
from plot import plot
from tqdm import tqdm
import os
import datetime
import multiprocessing as mp
import argparse
from tools import generate_givens_wall


# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = True
print_model_summary = True
print_policy = True
averaging_window = 5000

state_bounds = 1
gamma = 1
anil = 0.25
start = 1

save_length = True


def run(len_state=2, prob_1=3/14, prob_2=11/14, n_episodes = 250000, n_holes = 5, max_steps = 6, batch_size = 10, env_name="FoxInAHolev2",
        learning_rate = 0.001, n_hidden_layers = 2, n_nodes_per_layer = 2, exp_key = "default", givens_wall = None):
    
    input_dim = len_state
    activation = 'elu'
    n_actions = n_holes
    import tensorflow as tf
    from REINFORCE import reinforce_agent
    from NN import PolicyModel
    agent = reinforce_agent(batch_size=batch_size)
    model = PolicyModel(activation_function=activation, n_hidden_layers= n_hidden_layers, n_nodes_per_layer=n_nodes_per_layer, input_dim= input_dim, output_dim=n_actions, learning_rate= learning_rate)
    episode_reward_history = []
    episode_length_history = []
        
    for batch in tqdm(range(n_episodes // batch_size)):
        # Gather episodes
        
        episodes = agent.gather_episodes(state_bounds, n_holes, n_actions, model, batch_size, env_name, len_state=len_state, max_steps = max_steps, prob_1=prob_1, prob_2=prob_2, givens_wall= givens_wall)

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
    
    final_episodes = agent.gather_episodes(state_bounds, n_holes, n_actions, model, 1000, env_name, len_state=len_state, max_steps = max_steps, prob_1=prob_1, prob_2=prob_2, givens_wall= givens_wall)
    final_rewards = [ep['rewards'] for ep in final_episodes]
    final_episode_lengths= []

    for ep_rwds in final_rewards:
        final_episode_lengths.append(len(ep_rwds))

    avg_final_episode_lengths = np.mean(final_episode_lengths)
    std_final_episode_lengths = np.std(final_episode_lengths)

    print("The final performance averaged over 1000 episodes was {} +- {}.".format(avg_final_episode_lengths, std_final_episode_lengths))

    if print_policy:
        state = tf.convert_to_tensor([-1*np.ones(len_state)])
        policy = []
        for _ in range(2*(n_holes)):
            action = np.random.choice(n_holes, p = model(state).numpy()[0])
            policy.append(action)
            ar_state = state.numpy()
            ar_state[0] = np.roll(ar_state[0], 1)
            ar_state[0][0] = action
            state = tf.convert_to_tensor(ar_state)

        print("Final policy is the following sequence: {}".format(policy))


    if print_model_summary:
        model.model.summary()

    if plotting:
        plot(episode_reward_history, "NN", averaging_window)

    
    if save_data:

        # the path to where we save the results. we take the first letter of every _ argument block to determine this path
        # directory = f"/data1/bosman/resultsQRL/NN/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}/'
        
        # first save the trained model
        directory = f"/home/s2025396/data1/resultsQRL/NEW/NN/saved_models/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}"+f"{activation}/"

        if not os.path.isdir(directory):
            os.mkdir(directory)

        model.model.save(directory)
        
        # Save episode lengths 
        # Alice
        directory = f"/home/s2025396/data1/resultsQRL/NEW/NN/episode_lengths/"+exp_key+f'{n_holes}holes'+f'{n_hidden_layers}layers'+f''+f'{n_nodes_per_layer}nodes'+f'lr{learning_rate}'+f'neps{n_episodes}'+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}"+f"{activation}/"
        
        ## workstation
        # directory = f"/data1/bosman/resultsQRL/NN/ep_lengths/"+exp_key+f"{n_holes}holes"+f"{n_hidden_layers}layers"+f"{n_nodes_per_layer}nodes"+f"lr{learning_rate}"+f"neps{n_episodes}"+f"bsize{batch_size}"+f"gamma{gamma}"+f"start{start}"+f"anil{anil}"+f"{activation}/"
        if not os.path.isdir(directory):
            os.mkdir(directory)

        print(f"Storing results in {directory}")

        # add date and time to filename to create seperate files with the same setting.
        dt = str(datetime.datetime.now()).split()
        time = f";".join(dt[1].split(":"))

        directory += f"-".join((dt[0], time)) + ".npy"


        np.save(directory, episode_length_history)



def test_run():
    for i in tqdm(range(1000)):
        x = i*i

if __name__ == '__main__':   
    argparser = argparse.ArgumentParser()  
    argparser.add_argument("--len_state", "-ls", type = int, default= 2, help="The length of the input state.")
    argparser.add_argument("--prob_1", "-p1", type = float, default= 3/14, help="Probability of split to the right for QFIAH.")
    argparser.add_argument("--n_episodes", "-ne", type = int, default= 250000, help="Amount of episodes to train for.")
    argparser.add_argument("--n_holes", "-nh", type = int, default= 5, help="The amount of holes in the game.")
    argparser.add_argument("--max_steps", "-ms", type = int, default= 6, help="The amount of steps allowed within an episode.")
    argparser.add_argument("--batch_size", "-bs", type = int, default= 10, help="The batch size.")
    argparser.add_argument("--env_name", "-en", type = str, default= "FoxInAHolev2", help="The environment/game version.")
    argparser.add_argument("--learning_rate", "-lr", type = float, default= 0.001, help="The learning rate.")
    argparser.add_argument("--n_hidden_layers", "-nhl", type = int, default= 2, help="The amount of hidden layers.")
    argparser.add_argument("--n_nodes_per_layer", "-nnpl", type = int, default= 2, help="The amount of nodes per hidden layer.")
    argparser.add_argument("--n_reps","-nr", type = int, default= 10, help = "The amount of repetitions to run.")
    argparser.add_argument("--brick1", "-b1", type = str, default="gx", help="The type of brick for the 1st brick in the Givens wall. Note that due ot rules of matrix multiplication, the 2nd brick comes first in the wall.")
    argparser.add_argument("--theta1", "-t1", type = float, default=0.25, help="The rotation parameter for the 1st brick in the Givens wall. Will be mutliplied with pi.")
    argparser.add_argument("--brick2", "-b2", type = str, default="gy", help="The type of brick for the 2nd brick in the Givens wall.")
    argparser.add_argument("--theta2", "-t2", type = float, default=0.25, help="The rotation parameter for the 2nd brick in the Givens wall. Will be muliplied with pi.")

    args = argparser.parse_args()
    len_state = args.len_state
    prob_1= args.prob_1
    prob_2= 1-prob_1
    n_episodes = args.n_episodes
    n_holes = args.n_holes
    max_steps = args.max_steps
    batch_size = args.batch_size
    env_name = args.env_name
    learning_rate = args.learning_rate
    n_hidden_layers = args.n_hidden_layers
    n_nodes_per_layer = args.n_nodes_per_layer
    n_reps = args.n_reps
    brick1= args.brick1
    theta1= args.theta1 *np.pi
    brick2 = args.brick2
    theta2 = args.theta2 * np.pi

    givens_wall = None
 
    try:
        n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']
        print("cores provided are: "+f"{n_cores}")
        n_cores = int(n_cores)

    except:
        
        n_cores = n_reps

    print("The number of cores available is {}".format(n_cores))
    
    if env_name.lower()== 'qfiahv1'or env_name.lower()=='qfiahv2':

        exp_key = f"{len_state}inp-{env_name}-prob1{round(prob_1,2)}-prob2{round(prob_2,2)}-maxsteps{max_steps}"

    elif env_name.lower()== "givens":
        givens_wall = generate_givens_wall(n_holes, brick1, theta1, brick2, theta2)
        exp_key = f"{len_state}inp-{env_name}-{brick1}{round(theta1,2)}-{brick2}{round(theta2,2)}-maxsteps{max_steps}"

    else:
        exp_key = f"{len_state}inp-{env_name}-maxsteps{max_steps}"

    print("(len_state, prob_1, prob_2, n_episodes, n_holes, max_steps, batch_size, env_name, learning_rate, n_hidden_layers, n_nodes_per_layer, exp_key, brick1, theta1, brick2, theta2)")
    print(len_state, prob_1, prob_2, n_episodes, n_holes, max_steps, batch_size, env_name, learning_rate, n_hidden_layers, n_nodes_per_layer, exp_key, brick1, theta1, brick2, theta2)
    p = mp.Pool(int(n_cores))
    res = p.starmap(run, [(len_state, prob_1, prob_2, n_episodes, n_holes, max_steps, batch_size, env_name, learning_rate, n_hidden_layers, n_nodes_per_layer, exp_key, givens_wall) for _ in range(n_reps)])
    p.close()
    p.join()
    
