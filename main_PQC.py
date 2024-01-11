import numpy as np
from plot import plot
from tqdm import tqdm
import datetime
from time import time
import os
import multiprocessing as mp

import argparse
from tools import retrieve_transfer_matrices, write_results_to_pickle
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
from bruteforce_policy import run_guesses_givens, run_guesses_fiah


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = True
print_model_summary = True

state_bounds = 1
gamma = 1
averaging_window = 5000


anil = 0.25
start = 1


# Start training the agent
# for _ in range(n_reps):
def run(
    len_state=2,
    prob_1=3 / 14,
    tunneling_prob=0.2,
    n_episodes=250000,
    n_holes=5,
    max_steps=6,
    batch_size=10,
    env_name="FIAH",
    lr_in=0.01,
    lr_var=0.01,
    lr_out=0.01,
    n_layers=2,
    n_qubits=2,
    RxCnot=True,
    exp_key="default",
):
    from REINFORCE import reinforce_agent

    # from PQC import generate_model_policy, reinforce_update
    from PQC import generate_model_policy, reinforce_update
    import tensorflow as tf

    transfer_matrices = retrieve_transfer_matrices(
        env_name, n_holes, prob_1, tunneling_prob, brick1, theta1, brick2, theta2
    )

    agent = reinforce_agent(
        env_name, batch_size, max_steps, len_state, transfer_matrices
    )

    episode_reward_history = []
    episode_length_history = []
    # As the different sets of parameters require different learning rates, create seperate optimizers
    optimizer_in = tf.keras.optimizers.Adam(learning_rate=lr_in, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=lr_var, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=lr_out, amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2
    # w_in, w_var = 1, 0

    optimizers = [optimizer_in, optimizer_var, optimizer_out]
    ws = [w_in, w_var, w_out]

    model = generate_model_policy(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_actions=n_qubits,
        n_inputs=len_state,
        beta=1,
        RxCnot=RxCnot,
    )
    n_batches = n_episodes // batch_size

    for batch in tqdm(range(n_batches)):
        # # Gather episodes

        episodes, model_returns_nan = agent.gather_episodes_training(n_holes, model)

        if model_returns_nan:
            print(
                "The model output nans after about {} episodes".format(
                    batch * batch_size
                )
            )
            break

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep["states"] for ep in episodes])
        actions = np.concatenate([ep["actions"] for ep in episodes])
        rewards = [ep["rewards"] for ep in episodes]
        returns = np.concatenate(
            [agent.compute_returns(ep_rwds, gamma) for ep_rwds in rewards]
        )

        returns = np.array(returns, dtype=np.float32)

        eta = max(start * (1 - (batch * batch_size) / (n_episodes * anil)), 0)
        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(
            states,
            id_action_pairs,
            returns,
            model,
            ws,
            optimizers,
            batch_size=batch_size,
            eta=eta,
        )

        # Store collected rewards
        for ep_rwds in rewards:
            # if the fox was not found in the max amount of steps, act as if it is found in the next step.
            if ep_rwds[-1] == 0:
                episode_length_history.append(len(ep_rwds))
            else:
                episode_length_history.append(len(ep_rwds) + 1)

        if print_avg:
            avg_rewards = np.mean(episode_reward_history[-averaging_window:])
            print(
                "Finished episode",
                (batch + 1) * batch_size,
                "Average rewards: ",
                avg_rewards,
            )

    if model_returns_nan:
        return {
        "avg_performance": 0,
        "std": 0,
        "policy_avg": 0,
        "policy": 0
    }

    if plotting:
        plot(episode_reward_history, "PQC", averaging_window)

    begin = time()

    final_episodes, model_returns_nan = agent.gather_episodes_greedy(model, 1000)
    final_rewards = [ep["rewards"] for ep in final_episodes]
    final_episode_lengths = []

    for ep_rwds in final_rewards:
        final_episode_lengths.append(len(ep_rwds))

    avg_final_episode_lengths = np.mean(final_episode_lengths)
    std_final_episode_lengths = np.std(final_episode_lengths)

    end = time()

    print(
        "The final performance averaged over 1000 episodes was {} +- {}.".format(
            avg_final_episode_lengths, std_final_episode_lengths
        )
    )
    print("Determining the final performance took {} seconds".format(end - begin))


    state = tf.convert_to_tensor([-1 * np.ones(len_state)])
    policy = []
    for _ in range(max_steps):
        # action = np.random.choice(n_holes, p = model(state).numpy()[0])
        action = np.argmax(model(state).numpy()[0])
        policy.append(action)
        ar_state = state.numpy()
        ar_state[0] = np.roll(ar_state[0], 1)
        ar_state[0][0] = action
        state = tf.convert_to_tensor(ar_state)

    print("Final policy is the following sequence: {}".format(policy))
    if env_name.lower() == "fiah":
        avg = run_guesses_fiah(
            np.ones(n_holes) / n_holes, transfer_matrices, policy, max_steps
        )
    elif env_name.lower() == "givens":
        avg = run_guesses_givens(
            np.ones(n_holes) / np.linalg.norm(np.ones(n_holes)),
            transfer_matrices,
            policy,
            max_steps,
        )

    else:
        raise ValueError("Unimplemented environment.")

    print(f"The final policy gives a mathematical final performance of {avg:.4f}.")

    if save_data:
        # first save the trained model weights
        directory = (
            f"/home/s2025396/data1/resultsQRL/NEW/PQC/saved_models/"
            + exp_key
            + f"-nh{n_holes}"
            + f"-nq{n_qubits}"
            + f"-nl{n_layers}"
            + f"-ne{n_episodes}"
            + f"-lrin{lr_in}"
            + f"-lrvar{lr_var}"
            + f"-lrout{lr_out}"
            + f"-bs{batch_size}"
            + f"-start{start}anil{anil}/"
        )

        if not os.path.isdir(directory):
            os.mkdir(directory)

        dt = str(datetime.datetime.now()).split()
        current_time = f";".join(dt[1].split(":"))

        directory += f"-".join((dt[0], current_time))

        model.save_weights(directory)

        # the path to where we save the results. we take the first letter of every _ argument block to determine this path

        # ALICE
        directory = (
            f"/home/s2025396/data1/resultsQRL/NEW/PQC/episode_lengths/"
            + exp_key
            + f"-nh{n_holes}"
            + f"-nq{n_qubits}"
            + f"-nl{n_layers}"
            + f"-ne{n_episodes}"
            + f"-lrin{lr_in}"
            + f"-lrvar{lr_var}"
            + f"-lrout{lr_out}"
            + f"-bs{batch_size}"
            + f"-start{start}anil{anil}/"
        )

        ## WORKSTATION
        # directory = f"/data1/bosman/resultsQRL/PQC/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"

        if not os.path.isdir(directory):
            os.mkdir(directory)

        print(f"Storing results in {directory}")

        # add date and time to filename to create seperate files with the same setting.
        dt = str(datetime.datetime.now()).split()
        current_time = f";".join(dt[1].split(":"))

        directory += f"-".join((dt[0], current_time)) + ".npy"

        np.save(directory, episode_length_history)

    if print_model_summary:
        model.summary()

    return {
        "avg_performance": avg_final_episode_lengths,
        "std": std_final_episode_lengths,
        "policy_avg": avg,
        "policy": policy
    }


def test_run():
    for i in tqdm(range(1000)):
        x = i * i


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--len_state", "-ls", type=int, default=2, help="The length of the input state."
    )
    argparser.add_argument(
        "--prob_1",
        "-p1",
        type=float,
        default=3 / 14,
        help="Probability of split to the right for QFIAH.",
    )
    argparser.add_argument(
        "--tunneling_prob",
        "-tp",
        type=float,
        default=0.2,
        help="The tunneling probability for the QFIAH environment.",
    )
    argparser.add_argument(
        "--n_episodes",
        "-ne",
        type=int,
        default=250000,
        help="Amount of episodes to train for.",
    )
    argparser.add_argument(
        "--n_holes", "-nh", type=int, default=5, help="The amount of holes in the game."
    )
    argparser.add_argument(
        "--max_steps",
        "-ms",
        type=int,
        default=6,
        help="The amount of steps allowed within an episode.",
    )
    argparser.add_argument(
        "--batch_size", "-bs", type=int, default=10, help="The batch size."
    )
    argparser.add_argument(
        "--env_name",
        "-en",
        type=str,
        default="FIAH",
        help="The environment/game version.",
    )
    argparser.add_argument(
        "--lr_in",
        "-li",
        type=float,
        default=0.01,
        help="The learning rate for the input scaling.",
    )
    argparser.add_argument(
        "--lr_var",
        "-lv",
        type=float,
        default=0.01,
        help="The learning rate for the variational scaling.",
    )
    argparser.add_argument(
        "--lr_out",
        "-lo",
        type=float,
        default=0.01,
        help="The learning rate for observable weights.",
    )
    argparser.add_argument(
        "--n_layers", "-nl", type=int, default=2, help="The amount of layers."
    )
    argparser.add_argument(
        "--RxCnot",
        "-Rx",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use the PQC with RxCnot architecture or not. 0= True, 1= False",
    )
    argparser.add_argument(
        "--n_reps",
        "-nr",
        type=int,
        default=10,
        help="The amount of repetitions to run.",
    )
    argparser.add_argument(
        "--brick1",
        "-b1",
        type=str,
        default="gx",
        help="The type of brick for the 1st brick in the Givens wall. Note that due ot rules of matrix multiplication, the 2nd brick comes first in the wall.",
    )
    argparser.add_argument(
        "--theta1",
        "-t1",
        type=float,
        default=0.25,
        help="The rotation parameter for the 1st brick in the Givens wall. Will be mutliplied with pi.",
    )
    argparser.add_argument(
        "--brick2",
        "-b2",
        type=str,
        default="gy",
        help="The type of brick for the 2nd brick in the Givens wall.",
    )
    argparser.add_argument(
        "--theta2",
        "-t2",
        type=float,
        default=0.25,
        help="The rotation parameter for the 2nd brick in the Givens wall. Will be muliplied with pi.",
    )
    args = argparser.parse_args()
    len_state = args.len_state
    prob_1 = args.prob_1
    tunneling_prob = args.tunneling_prob
    n_episodes = args.n_episodes
    n_holes = args.n_holes
    max_steps = args.max_steps
    batch_size = args.batch_size
    env_name = args.env_name
    lr_in = args.lr_in
    lr_var = args.lr_var
    lr_out = args.lr_out
    n_layers = args.n_layers
    n_qubits = n_holes
    RxCnot = args.RxCnot
    brick1 = args.brick1
    theta1 = args.theta1 * np.pi
    brick2 = args.brick2
    theta2 = args.theta2 * np.pi

    if RxCnot == 0:
        RxCnot = False
    elif RxCnot == 1:
        RxCnot = True
    n_reps = args.n_reps

    if env_name.lower() == "qfiah":
        if RxCnot:
            exp_key = f"{len_state}inp-{env_name}-p1{round(prob_1,2)}-tp{tunneling_prob}-ms{max_steps}-PQC-RxCNOT"

        else:
            exp_key = f"{len_state}inp-{env_name}-p1{round(prob_1,2)}-tp{tunneling_prob}-ms{max_steps}-PQC-"

        specific_env_id = env_name + f"-p1{round(prob_1,2)}-tp{tunneling_prob}"

    elif env_name.lower() == "givens":
        if RxCnot:
            exp_key = f"{len_state}inp-{env_name}-{brick1}{round(theta1,2)}-{brick2}{round(theta2,2)}-ms{max_steps}-PQC-RxCNOT"

        else:
            exp_key = f"{len_state}inp-{env_name}-{brick1}{round(theta1,2)}-{brick2}{round(theta2,2)}-ms{max_steps}-PQC"

        specific_env_id = (
            env_name + f"-{brick1}{round(theta1,2)}-{brick2}{round(theta2,2)}"
        )
    else:
        if RxCnot:
            exp_key = f"{len_state}inp-{env_name}-ms{max_steps}-PQC-RxCnot"

        else:
            exp_key = f"{len_state}inp-{env_name}-ms{max_steps}-PQC"

        specific_env_id = env_name

    # run(len_state, prob_1, prob_2, n_episodes, n_holes, max_steps, batch_size, env_name, lr_in, lr_var, lr_out, n_layers, n_qubits, RxCnot, exp_key)

    print(
        "Settings: len_state {}, prob_1 {}, n_episodes {}, n_holes {}, max_steps {}, batch_size {}, env_name {}, lr_in {}, lr_var {}, lr_out {}, n_layers {}, n_qubits {}, RxCnot {}, exp_key {}, brick1 {}, theta1 {}, brick2 {}, theta2 {}".format(
            len_state,
            prob_1,
            n_episodes,
            n_holes,
            max_steps,
            batch_size,
            env_name,
            lr_in,
            lr_var,
            lr_out,
            n_layers,
            n_qubits,
            RxCnot,
            exp_key,
            brick1,
            theta1,
            brick2,
            theta2,
        )
    )

    # run(len_state, prob_1, prob_2, 250, n_holes, max_steps, batch_size, env_name, lr_in, lr_var, lr_out, n_layers, n_qubits, RxCnot, exp_key, givens_wall)

    p = mp.Pool(n_reps)
    res = p.starmap(
        run,
        [
            (
                len_state,
                prob_1,
                tunneling_prob,
                n_episodes,
                n_holes,
                max_steps,
                batch_size,
                env_name,
                lr_in,
                lr_var,
                lr_out,
                n_layers,
                n_qubits,
                RxCnot,
                exp_key,
            )
            for _ in range(n_reps)
        ],
    )
    p.close()
    p.join()

    results_dict = {}
    for key in res[0].keys():
        results_dict[key] = np.array([d[key] for d in res])

    write_results_to_pickle(results_dict= results_dict, len_state = len_state, n_holes= n_holes, type= "PQC", specific_env_id= specific_env_id, 
                         n_layers= n_layers, RxCnot = RxCnot,
                         n_episodes= n_episodes, learning_rate= lr_var, max_steps= max_steps, 
                         batch_size=batch_size, n_reps=n_reps)
    
    print("Results have been succesfully pickled")
