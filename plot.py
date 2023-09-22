# This file wil allow us to create smoothed plots of the average episode rewards over time
import os

import matplotlib.pyplot as plt
import numpy as np

def plot(episode_reward_history, label, smoothing):
    """
    This functions will plot the average episode reward over a ~ thousand budget step window. It takes as input a folder directory containing
    .npy files with the data. Averages will be taken across these files to create statistically meaningfull results.
    """
    # os.listdir
    counter = 0
    n_episodes = 0
    smoothed_episode_rewards = []
    for episode_reward in episode_reward_history:

        counter += episode_reward
        n_episodes += 1

        if n_episodes == smoothing:
            smoothed_episode_rewards.append(counter / n_episodes)
            counter = 0
            n_episodes = 0


    # X = np.linspace(0, len(episode_reward_history)*smoothing, len(episode_reward_history)/smoothing)
    X = np.arange(smoothing, (len(smoothed_episode_rewards)+1)*smoothing, smoothing)

    plt.plot(X, smoothed_episode_rewards, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Average episode reward")
    plt.show()

    pass

def plot_reward_from_path(path_folders, labels, smoothing):
    """
    This functions will plot the average episode reward over a ~ thousand budget step window. It takes as input a folder directory containing
    .npy files with the data. Averages will be taken across these files to create statistically meaningfull results.
    """
    # os.listdir
    for p, path_folder in enumerate(path_folders):
        n_files = 0
        for path in os.listdir(path_folder):
            n_files += 1

        all_data = np.zeros((n_files,
                             0)).tolist()  # [[] for n in range(n_files)] # creates a list of empty lists for the amount of different data files available
        min_length = 10000000  # min length was to keep track of issues

        for i, path in enumerate(os.listdir(path_folder)):
            dir = os.path.join(path_folder, path)
            episode_rewards = np.load(
                dir)  # now names episode_rewards, but depending on dir can also be episode_lengths
            counter = 0
            n_episodes = 0
            smoothed_episode_rewards = []
            for episode in episode_rewards:

                counter += episode
                n_episodes += 1

                if n_episodes == smoothing:
                    smoothed_episode_rewards.append(counter / n_episodes)
                    counter = 0
                    n_episodes = 0

            min_length = min(min_length, len(smoothed_episode_rewards))
            all_data[i] = smoothed_episode_rewards
        for i, array in enumerate(all_data):
            all_data[i] = array[:min_length]

        all_data = np.asarray(all_data)
        all_data_std = np.std(all_data, axis=0)
        all_data_average = np.average(all_data, axis=0)

        X = np.linspace(0, min_length * smoothing, min_length)

        plt.plot(X, all_data_average, label=labels[p])

        if n_files > 1:
            all_data_std = np.std(all_data, axis=0)
            plt.fill_between(X, all_data_average - all_data_std, all_data_average + all_data_std, alpha=0.2)

        plt.xlabel("Episode")
        plt.ylabel("Average episode reward")

    pass

def plot_length_from_path(path_folders, labels, smoothing):
    """
    This functions will plot the average episode reward over a ~ thousand budget step window. It takes as input a folder directory containing
    .npy files with the data. Averages will be taken across these files to create statistically meaningfull results.
    """
    # os.listdir
    for p, path_folder in enumerate(path_folders):
        n_files = 0
        for path in os.listdir(path_folder):
            n_files += 1

        all_data = np.zeros((n_files,
                             0)).tolist()  # [[] for n in range(n_files)] # creates a list of empty lists for the amount of different data files available
        min_length = 10000000  # min length was to keep track of issues

        for i, path in enumerate(os.listdir(path_folder)):
            dir = os.path.join(path_folder, path)
            episode_lengths = np.load(
                dir)  # now names episode_rewards, but depending on dir can also be episode_lengths
            counter = 0
            n_episodes = 0
            smoothed_episode_lenghts = []
            for episode in episode_lengths:

                counter += episode
                n_episodes += 1

                if n_episodes == smoothing:
                    smoothed_episode_lenghts.append(counter / n_episodes)
                    counter = 0
                    n_episodes = 0

            min_length = min(min_length, len(smoothed_episode_lenghts))
            all_data[i] = smoothed_episode_lenghts
        for i, array in enumerate(all_data):
            all_data[i] = array[:min_length]

        all_data = np.asarray(all_data)
        all_data_std = np.std(all_data, axis=0)
        all_data_average = np.average(all_data, axis=0)

        X = np.linspace(0, min_length * smoothing, min_length)

        plt.plot(X, all_data_average, label=labels[p])

        if n_files > 1:
            all_data_std = np.std(all_data, axis=0)
            plt.fill_between(X, all_data_average - all_data_std, all_data_average + all_data_std, alpha=0.2)

        plt.xlabel("Episode")
        plt.ylabel("Average episode lengths")

    pass


def plot_from_path_percentage(path_folders, labels, smoothing=1000):
    """
    This functions will plot the average episode reward over a ~ thousand budget step window. It takes as input a folder directory containing
    .npy files with the data. Averages will be taken across these files to create statistically meaningfull results.
    """
    # os.listdir
    for p, path_folder in enumerate(path_folders):
        n_files = 0
        for path in os.listdir(path_folder):
            n_files += 1

        all_data = []  # [[] for n in range(n_files)] # creates a list of empty lists for the amount of different
        # data files available

        for i, path in enumerate(os.listdir(path_folder)):
            dir = os.path.join(path_folder, path)
            rewards = np.load(dir)  # now names episode_rewards, but depending on dir can also be episode_lengths
            budget = len(rewards)
            n_windows = budget // smoothing

            smoothed_percentages = []

            for window in range(n_windows):
                smoothing_batch = rewards[window * smoothing:(window + 1) * smoothing]
                n_misses = np.sum(smoothing_batch == -1)
                n_catches = np.sum(smoothing_batch == 1)
                total_opportunities = n_misses + n_catches
                smoothed_percentages.append(n_catches / total_opportunities)

            all_data.append(smoothed_percentages)

        all_data = np.asarray(all_data)
        all_data_average = np.average(all_data, axis=0)

        X = np.arange(budget, step=smoothing)
        plt.plot(X, all_data_average, label=labels[p])

        if n_files > 1:
            all_data_std = np.std(all_data, axis=0)
            plt.fill_between(X, all_data_average - all_data_std, all_data_average + all_data_std, alpha=0.1)

        plt.xlabel("Budget")
        plt.ylabel("Catch ratio")

    pass


# adjust to own paths

if __name__ == '__main__':
    paths = [
        # "/data1/bosman/resultsQRL/NN/2(n-2)-inp-enc5holes2layers10nodeslr0.001neps100000bsize100/",
        # "/data1/bosman/resultsQRL/NN/2(n-2)-inp-enc5holes2layers10nodeslr0.005neps100000bsize100/",
        # 





        # "/data1/bosman/resultsQRL/NN/2(n-2)-inp-enc5holes2layers10nodeslr0.01neps100000bsize100elu/",
        # "/data1/bosman/resultsQRL/NN/2(n-2)-inp-enc5holes2layers10nodeslr0.01neps100000bsize100/"


        # ## For the baselines
        ## 5 holes 

        # "ProbabilityAgent10holes/"


        # ## 10 holes 
        # "BoundAgent10holes/",
        # "BoundAgentUnbounded10holes/",
        # "ProbabilityAgent10holes/",
        # "ProbabilityAgentUnbounded10holes/"

        # "lengthsProbabilityAgent10holes1000000neps/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes3layers50nodeslr0.0001neps1000000bsize10gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes3layers75nodeslr0.0001neps1000000bsize10gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes3layers100nodeslr0.0001neps1000000bsize10gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes4layers50nodeslr0.0001neps1000000bsize10gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes5layers50nodeslr0.0001neps1000000bsize10gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes6layers50nodeslr0.0001neps1000000bsize10gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes3layers100nodeslr0.0001neps1000000bsize100gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes3layers100nodeslr0.001neps1000000bsize100gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes3layers100nodeslr0.0001neps1000000bsize20gamma1start1anil0.25elu/",
        # "/data1/bosman/resultsQRL/NN/ep_length/2(n-2)-inp-enc-rewardfound110holes3layers100nodeslr0.001neps1000000bsize10gamma1start1anil0.25elu/"

        # PQC
        "lengthsProbabilityAgent5holes/",
        "/data1/bosman/resultsQRL/PQC/ep_length/2(n-2)-inp-enc-rewardfound15holes8layersneps100000lrin0.1lrvar0.01lrout0.1bsize10gamma1start1anil0.25/"
    ]
    labels = [

            # "probability agent 10 holes",
            # "lr 0.0001 50 nodesn 3 hl",
            # "lr 0.0001 75 nodes 3 hl ",
            # "lr 0.0001 100 nodes 3 hl",
            # "lr 0.0001 50 nodes 4 hl",
            # "lr 0.0001 50 nodes 5 hl",
            # "lr 0.0001 50 nodes 6 hl",
            # "lr 0.0001 bs 100",
            # "lr 0.001 bs 100",
            # "lr 0.0001 bs 20",
            # "lr 0.001 bs 10"


        # ## For the baselines
        # ## 10 holes
        # "Bound agent 10 holes",
        # "Bound agent unbounded 10 holes",
        # "Probability agent 10 holes",
        # "Probability agent unbounded 10 holes "
        # "1 hidden layer",
        # "2 hidden layers",
        # "3 hidden layers",
        # "4 hidden layers"

        # "1 hidden layer",
        # "2 hidden layers",
        # "3 hidden layers",
        # "4 hidden layers"
        "Probability agent 5 holes",
        "PQC 8 layers"

    ]


    plot_length_from_path(paths, labels, 5000)
    plt.legend()
    plt.show()