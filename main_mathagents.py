import numpy as np
from plot import plot
from tqdm import tqdm
import os
import datetime
from mathematical_agents import ProbabilityAgent, BoundAgent, PickMiddle, QuantumProbabilityAgent
import multiprocessing as mp

# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = True
plot_probabilities = False

from fox_in_a_hole_gym import FoxInAHolev2, QFIAHv1, QFIAHv2

exp_key = "ProbabilityAgent"
env_name = "FoxInAHolev2"
max_steps = 16
prob1= 3/14
prob2= 11/14
tunneling_prob= 0
n_episodes =1000000
n_holes = 10
n_actions = n_holes
n_reps = 10

averaging_window = 5000

def run():
    if exp_key == "ProbabilityAgent":
        agent = ProbabilityAgent(n_holes = n_holes, print_hole_prob= plot_probabilities)

    elif exp_key == "ProbabilityAgentUnbounded":
        agent = ProbabilityAgent(n_holes = n_holes, print_hole_prob= plot_probabilities)

    elif exp_key =="BoundAgent":
        agent = BoundAgent(n_holes=n_holes)

    elif exp_key =="BoundAgentUnbounded":
        agent = BoundAgent(n_holes=n_holes)

    elif exp_key=="PickMiddle":
        agent = PickMiddle(n_holes=n_holes)

    elif exp_key == "QuantumProbabilityAgent":
        agent = QuantumProbabilityAgent(n_holes =n_holes, print_holeprob=plot_probabilities, prob_1=prob1,prob_2=prob2, tunneling_prob= tunneling_prob)


    else:
        raise KeyError

    if plot_probabilities:
        for _ in range(10):
            agent.pick_hole()
            agent.update_probabilities()
                
    # Start training the agent
    else:
        pass

    if env_name == "FoxInAHolev2":
        env = FoxInAHolev2(n_holes=n_holes, len_state=2, max_steps = max_steps)
    elif env_name == "QFIAHv1":
        env = QFIAHv1(n_holes, len_state = 2, max_steps = 6, prob_1 = prob1, prob_2= prob2)
    elif env_name == "QFIAHv2":
        env = QFIAHv2(n_holes, len_state = 2, max_steps = 6, prob_1 = prob1, prob_2= prob2, tunneling_prob= tunneling_prob)
    else:
        raise ValueError

    episode_reward_history = []
    episode_length_history = []
    for _ in tqdm(range(n_episodes)):
        done = False
        ep_rwds = []
        env.reset()
        while not done:

            action = agent.pick_hole()
            if exp_key == "ProbabilityAgentUnbounded" or exp_key == "BoundAgentUnbounded":
                state, reward, done, _ = env.unbounded_step(action)
            
            else:
                state, reward, done, _ = env.step(action)
            ep_rwds.append(reward)

            try:
                agent.update_probabilities()
            except:
                pass
        
        episode_reward_history.append(np.sum(ep_rwds))
        episode_length_history.append(len(ep_rwds))
        agent.reset()


    print("the policy followed is {}".format(agent.longest_policy_sequence))

    if plotting:
        plot(episode_reward_history, "NN", averaging_window)

    if save_data:

        # the path to where we save the results. we take the first letter of every _ argument block to determine this path
        directory = f'./rewards{env_name}env{exp_key}{n_holes}holes{n_episodes}neps{max_steps}steps{round(prob1,2)}prob1{round(prob2,2)}prob2{tunneling_prob}tunnelingprob/'

        if not os.path.isdir(directory):
            os.mkdir(directory)

        print(f"Storing results in {directory}")

        # add date and time to filename to create seperate files with the same setting.
        dt = str(datetime.datetime.now()).split()
        time = f";".join(dt[1].split(":"))

        directory += f"-".join((dt[0], time)) + ".npy"


        np.save(directory, episode_reward_history)
    
    if save_data:

        # the path to where we save the results. we take the first letter of every _ argument block to determine this path
        directory = f'./lengths{env_name}env{exp_key}{n_holes}holes{n_episodes}neps{max_steps}steps{round(prob1,2)}prob1{round(prob2,2)}prob2{tunneling_prob}tunnelingprob/'

        if not os.path.isdir(directory):
            os.mkdir(directory)

        print(f"Storing results in {directory}")

        # add date and time to filename to create seperate files with the same setting.
        dt = str(datetime.datetime.now()).split()
        time = f";".join(dt[1].split(":"))

        directory += f"-".join((dt[0], time)) + ".npy"


        np.save(directory, episode_length_history)

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


            

