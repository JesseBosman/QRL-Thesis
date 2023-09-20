import numpy as np
from plot import plot
from tqdm import tqdm
import os
import datetime
from mathematical_agents import ProbabilityAgent, BoundAgent

# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = True
plot_probabilities = False

from fox_in_a_hole_gym import FoxInAHole


exp_key = "ProbabilityAgentUnbounded"
n_episodes = 100000
n_holes = 5
n_actions = n_holes
n_reps = 5

averaging_window = 5000
if exp_key == "ProbabilityAgent":
    agent = ProbabilityAgent(n_holes = n_holes, print_hole_prob= plot_probabilities)

elif exp_key == "ProbabilityAgentUnbounded":
    agent = ProbabilityAgent(n_holes = n_holes, print_hole_prob= plot_probabilities)

elif exp_key =="BoundAgent":
    agent = BoundAgent(n_holes=n_holes)

elif exp_key =="BoundAgentUnbounded":
    agent = BoundAgent(n_holes=n_holes)


else:
    raise KeyError

if plot_probabilities:
    for _ in range(10):
        agent.pick_hole()
        agent.update_probabilities()
            
# Start training the agent
else:
    env = FoxInAHole(n_holes=n_holes)
    for rep in range(n_reps):
        episode_reward_history = []
        env.reset()
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
            agent.reset()


        print("the policy followed is {}".format(agent.longest_policy_sequence))

        if plotting:
            plot(episode_reward_history, "NN", averaging_window)

        if save_data:

            # the path to where we save the results. we take the first letter of every _ argument block to determine this path
            directory = f'./{exp_key}{n_holes}holes/'

            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_reward_history)

