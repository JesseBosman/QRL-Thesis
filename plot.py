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

def plot_from_path(path_folders, labels, smoothing):
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
        plt.ylabel("Average episode reward/length")

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


# paths = [
#     # # "results/-NN5holeslayers2nodes20lr0.005/",
#     # "results/-NN10holeslayers2nodes20lr0.005/"
#     "data1/bosman/resultsQRL/-NN5holeslayers5nodeslrin0.1lrlrvar0.01lrout0.1/"
# ]
# labels = [
#     # "NN 5 holes",
#     # "NN 10 holes"
#     "PQC"
# ]


# plot_from_path(paths, labels, 5000)
# plt.legend()
# # plt.show()