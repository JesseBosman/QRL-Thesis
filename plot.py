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

def plot_length_from_path(path_folders, labels, smoothing, neps, alpha = 0.1, mean = True):
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
                dir)[:neps]  # now names episode_rewards, but depending on dir can also be episode_lengths
            counter = 0
            n_episodes = 0
            # print("max length was {}".format(np.max(episode_lengths)))
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

        X = np.linspace(0, min_length * smoothing, min_length)

        if mean:
            all_data_std = np.std(all_data, axis=0)
            all_data_average = np.average(all_data, axis=0)
            plt.plot(X, all_data_average, label=labels[p])

            if n_files > 1:
                all_data_std = np.std(all_data, axis=0)
                plt.fill_between(X, all_data_average - all_data_std, all_data_average + all_data_std, alpha=alpha)
                plt.ylabel("Average episode lengths")
        
        else:
            for i in range(n_files):

                plt.plot(X, all_data[i], label = f"run {i}")
            
            plt.ylabel("Episode lengths")
        plt.xlabel("Episode")

        print("The amount of files averaged over for path {} was {}".format(p, n_files))
        

    pass


def plot_length_from_path_best(path_folders, labels, smoothing, n_eps, n_best = 1, alpha = 0.1):
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
                             0)).tolist() 

        for i, path in enumerate(os.listdir(path_folder)):
            dir = os.path.join(path_folder, path)
            episode_lengths = np.load(
                dir)[:n_eps]  
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

            all_data[i]= smoothed_episode_lenghts

        all_data = np.array(all_data)
        final_values = all_data[:,-1]
        index = np.argpartition(final_values, n_best)[:n_best]
        all_data = all_data[index]
        all_data_average = np.average(all_data, axis=0)
        X = np.linspace(0, len(smoothed_episode_lenghts)*smoothing, len(smoothed_episode_lenghts))
        if n_best > 1:
            all_data_std = np.std(all_data, axis=0)
            
            plt.plot(X, all_data_average, label=labels[p])
            all_data_std = np.std(all_data, axis=0)
            plt.fill_between(X, all_data_average - all_data_std, all_data_average + all_data_std, alpha=alpha)
            plt.ylabel("Average episode lengths")
        else: 
            plt.plot(X, all_data_average, label= labels[p])
            plt.ylabel("Episode lengths")
        print("The amount of files in path {} was {}".format(p, n_files))

    plt.xlabel("Episode")

    pass


# adjust to own paths
if __name__ == '__main__':
    paths = [
        "/home/s2025396/data1/resultsQRL/NEW/PQC/episode_lengths/2inp-fiah-ms10-PQC-RxCnot-nh5-nq5-nl1-ne2500-lrin0.01-lrvar0.01-lrout0.01-bs10-start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/NEW/PQC/episode_lengths/2inp-fiah-ms10-PQC-RxCnot-nh5-nq5-nl1-ne2500-lrin0.1-lrvar0.1-lrout0.1-bs10-start1anil0.25/",
    ]

    labels = [
        "2inpRxCnot1llre-2 (17)",
        "2inpRxCnot1llre-1 (17)",
    ]
    n_eps = 2500
    smoothing = 10

    plot_length_from_path_best(paths, labels, smoothing, n_eps, n_best=2)
    plt.legend()
    plt.savefig("plot_best.pdf")
    plt.close()

    plot_length_from_path(paths, labels, smoothing, n_eps, alpha = 0.1, mean= True)
    plt.legend()
    plt.savefig("plot_mean.pdf")
    plt.close()

    