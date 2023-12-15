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
        # ## Givens 6 holes gx 0.25 gy 0.5
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-Givens-gx0.79-gy1.57-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-Givens-gx0.79-gy1.57-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/3inp-Givens-gx0.79-gy1.57-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-Givens-gx0.79-gy1.57-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10006holes2layers5nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10006holes2layers5nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10006holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10006holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10006holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10006holes1layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits5layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits5layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits4layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits4layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits3layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits2layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",

        # # Givens 5 holes gx 0.25 gy 0.5
        
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-Givens-gx0.79-gy1.57-maxsteps10005holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-Givens-gx0.79-gy1.57-maxsteps10005holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-Givens-gx0.79-gy1.57-maxsteps10005holes2layers5nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-Givens-gx0.79-gy1.57-maxsteps10005holes2layers5nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",

        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/1inp-Givens-gx0.79-gy1.57-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-Givens-gx0.79-gy1.57-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/3inp-Givens-gx0.79-gy1.57-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-Givens-gx0.79-gy1.57-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy1.57-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",

        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",        
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits2layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits4layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits4layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits5layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-Givens-gx0.79-gy1.57-maxsteps1000-PQC-v4-RxCNOT5holes5qubits5layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        
        
        # ## QFIAHv2 6 holes
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/1inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/3inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes2layers4nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-QFIAHv2-prob10.21-prob20.79-maxsteps10006holes1layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/4-inp-QFIAHv2-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits5layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/4-inp-QFIAHv2-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/4-inp-QFIAHv2-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/"

        # ## Givens 5 holes theta1=theta2=0.25 block1=gx block2=gy
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits2layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits4layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits4layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits5layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-Givens-gx0.79-gy0.79-maxsteps1000-PQC-v4-RxCNOT5holes5qubits5layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes1layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes2layers3nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes2layers5nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes2layers5nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/1inp-Givens-gx0.79-gy0.79-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-Givens-gx0.79-gy0.79-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/3inp-Givens-gx0.79-gy0.79-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-Givens-gx0.79-gy0.79-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-Givens-gx0.79-gy0.79-maxsteps10005holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        
        
        # ## QFIAHv1 6 holes prob1=0.5
        # "lengthsQFIAHv1envQuantumProbabilityAgent6holes250000neps1000steps0.5prob10.5prob20tunnelingprob/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.5-prob20.5-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/6-inp-QFIAHv1-prob10.5-prob20.5-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.5-prob20.5-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"

        # QFIAHv1 6 holes prob1=3/14
        "lengthsQFIAHv1envQuantumProbabilityAgent6holes250000neps1000steps0.21prob10.79prob20tunnelingprob/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/6-inp-QFIAHv1-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/6-inp-QFIAHv1-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/6-inp-QFIAHv1-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/6-inp-QFIAHv1-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits4layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/6-inp-QFIAHv1-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits5layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes1layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes1layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/1inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/3inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/4inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/10inp-QFIAHv1-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",

        
        # # QFIAHv2 5 holes
        # "lengthsQFIAHv2envQuantumProbabilityAgent5holes250000neps6steps0.21prob10.79prob20.2tunnelingprob/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv2-maxsteps6-red_PQC-RxCNOT5holes3qubits1layersneps250000lr0.01bsize1gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes3layers10nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/"
        
       
        # # QFIAHv1 5 holes prob1!=prob2 (done)
        # "lengthsQuantumProbabilityAgent5holes500000neps6steps0.21prob10.79prob2",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-red_PQC-RxCNOT5holes3qubits1layersneps250000lr0.01bsize1gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps65holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"



        # # QFIAHv1 5 holes prob1=prob2 (done)
        # "lengthsQuantumProbabilityAgent5holes500000neps6steps0.5prob10.5prob2/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.5-prob20.5-maxsteps6-red_PQC-RxCNOT5holes3qubits1layersneps250000lr0.01bsize1gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-PQC-v4-RXCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-PQC-v45holes5qubits1layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-QFIAHv1-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.5-prob20.5-maxsteps65holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv15holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv15holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv15holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"

        # # Classical FIAH 10 holes
        # "lengthsFoxInAHolev2envProbabilityAgent10holes1000000neps16steps0.21prob10.79prob20tunnelingprob/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits1layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits1layersneps1000000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits1layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits1layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes1layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes1layers2nodeslr0.01neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes1layers2nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes2layers2nodeslr0.01neps500000bsize10gamma1start1anil0.25elu/",
        # #"/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes2layers2nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits2layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits2layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits3layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v410holes10qubits3layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits5layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits5layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits5layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits5layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes5layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers6nodeslr0.001neps1000000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers10nodeslr0.0001neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits10layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits10layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits20layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v410holes10qubits20layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        
        
        # Classical FIAH 6 holes
        # "lengthsProbabilityAgent6holes1000000neps1000steps0.21prob10.79prob2",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/6-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1000-PQC-v4-RxCNOT6holes6qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps10006holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps10006holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps10006holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps10006holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/6inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps10006holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"


        ## Classical FIAH 5 holes (done)
        # "lengthsProbabilityAgent5holes500000neps6steps0.5prob10.5prob2",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/3inp-FoxInAHolev25holes2layers3nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers10nodeslr0.0005neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers4nodeslr0.0005neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers4nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes2layers5nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes2layers3nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps65holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes1layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-PQC-v35holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/3-inp-PQC-v35holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes3qubits1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.1lrvar0.01lrout0.1bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.1lrvar0.1lrout0.1bsize10gamma1start1anil0.25/"
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-maxsteps6-red_PQC-RxCNOT5holes3qubits1layersneps250000lr0.01bsize1gamma1start1anil0.25/"
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-maxsteps6-red_PQC-RxCNOT5holes3qubits1layersneps250000lr0.01bsize1gamma1start1anil0.25/"
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT-noobsweights5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/"

    ]

       
    labels = [
        # ## Givens 6 holes gx 0.25 gy 0.5
        # # "6inpNN3l10nlre-3 (356)",
        # "5inpNN3l10nlre-3 (346)",
        # # "4inpNN3l10nlre-3 (336)",
        # # "3inpNN3l10nlre-3 (326)",
        # # "2inpNN3l10nlre-3 (316)",
        # # "5inpNN2l5nlre-2 (96)",
        # "5inpNN2l5nlre-3 (96)",
        # # "5inpNN2l2nlre-2 (36)",
        # "5inpNN2l2nlre-3 (36)",
        # "5inpNN1l2nlre-2 (30)",
        # # "5inpNN1l2nlre-3 (30)",

        # "5inpRxCnot5llr[e-2,e-2,e-2] (67)",
        # # "5inpRxCnot5llr[e-2,e-3,e-2] (67)",
        # # "5inpRxCnot4llr[e-2,e-2,e-2] (56)",
        # "5inpRxCnot4llr[e-2,e-3,e-2] (56)",
        # # "5inpRxCnot3llr[e-2,e-2,e-2] (45)",
        # "5inpRxCnot3llr[e-2,e-3,e-2] (45)",
        # # "5inpRxCnot2llr[e-2,e-2,e-2] (34)",
        # "5inpRxCnot2llr[e-2,e-3,e-2] (34)",
        # "5inpRxCnot1llr[e-2,e-2,e-2] (23)",
        # # "5inpRxCnot1llr[e-2,e-3,e-2] (23)",

        # # Givens 5 holes gx 0.25 gy 0.5 
        # "2inpNN1l2nlre-2 (21)",
        # "2inpNN2l2nlre-2 (27)",
        # # "2inpNN2l5nlre-2 (75)",
        # "2inpNN2l5nlre-3 (75)",

        # # "1inpNN3l10nlre-3 (355)",
        # "2inpNN3l10nlre-3 (355)",
        # # "3inpNN3l10nlre-3 (355)",
        # # "4inpNN3l10nlre-3 (355)",
        # # "5inpNN3l10nlre-3 (335)",
        
        # "2inpRxCnot1llr[e-2,e-2,e-2] (17)",
        # # "2inpRxCnot1llr[e-2,e-3,e-2] (17)",
        # # "2inpRxCnot2llr[e-2,e-2,e-2] (24)",
        # "2inpRxCnot2llr[e-2,e-3,e-2] (24)",
        # "2inpRxCnot3llr[e-2,e-2,e-2] (31)",
        # # "2inpRxCnot3llr[e-2,e-3,e-2] (31)",
        # "2inpRxRyRzCz1llr[e-2,e-2,e-2] (32)",
        # # "2inpRxCnot4llr[e-2,e-2,e-2] (38)",
        # "2inpRxCnot4llr[e-2,e-3,e-2] (38)",
        # # "2inpRxCnot5llr[e-2,e-2,e-2] (45)",
        # "2inpRxCnot5llr[e-2,e-3,e-2] (45)",


        # ## QFIAHv2 6 holes
        # # "1inpNN3l10nlre-3",
        # # "2inpNN3l10nlre-3",
        # # "3inpNN3l10nlre-3",
        # "4inpNN3l10nlre-3 (333)",
        # # "5inpNN3l10nlre-3",
        # # "6inpNN3l10nlre-3"
        # "4inpNN2l4nlre-3 (70)",
        # "4inpNN2l2nlre-3 (34)",
        # "4inpNN1l2nlre-3 (28)",
        # "4inpRxCnot5llr[e-2,e-2,e-2] (62)",
        # "4inpRxCnot3llr[e-2,e-2,e-2] (42)",
        # "4inpRxCnot1llr[e-2,e-2,e-2] (22)"

        # # ## Givens 5 holes theta1=theta2=0.25 block1=gx block2=gy
        # "5inpRxCnot1llr[e-2,e-2,e-2] (20)",
        # "5inpRxCnot1llr[e-2,e-3,e-2] (20)",
        # # "5inpRxCnot2llr[e-2,e-2,e-2] (30)",
        # # "5inpRxCnot2llr[e-2,e-3,e-2] (30)",
        # # "5inpRxCnot3llr[e-2,e-2,e-2] (40)",
        # # "5inpRxCnot3llr[e-2,e-3,e-2] (40)",
        # # "5inpRxCnot4llr[e-2,e-2,e-2] (50)",
        # # "5inpRxCnot4llr[e-2,e-3,e-2] (50)",
        # # "5inpRxCnot5llr[e-2,e-2,e-2] (60)",
        # # "5inpRxCnot5llr[e-2,e-2,e-2] (60)",
        # "5inpNN1l2nlre-2 (27)",
        # # "5inpNN1l2nlre-3 (27)",
        # # "5inpNN2l2nlre-2 (33)",
        # "5inpNN2l2nlre-3 (33)",
        # "5inpNN2l3nlre-3 (50)",
        # # "5inpNN2l5nlre-2 (90)",
        # "5inpNN2l5nlre-3 (90)",
        # # "1inpNN3l10nlre-3",
        # # "2inpNN3l10nlre-3",
        # # "3inpNN3l10nlre-3",
        # # "4inpNN3l10nlre-3",
        # "5inpNN3l10nlre-3 (335)",


        # ##QFIAHv1 6 holes prob1=0.5
        # "Probability Agent",
        # # "2inpRxCnot1llr[e-2,e-2,e-2] (20)",
        # "6inpRxCnot1llr[e-2,e-2,e-2] (24)",
        # # "2inpNN1l2nlre-2 (24)",
        # "6inpNN1l2nlre-2 (32)",
        # # "2inpNN2l2nlre-2 (30)",
        # # "2inpNN2l2nlre-3 (30)",
        # "6inpNN2l2nlre-2 (38)",
        # # "6inpNN2l2nlre-3 (38)",
        # # "2inpNN3l10nlre-3 (316)",
        # "6inpNN3l10nlre-3 (356)"

        #QFIAHv1 6 holes prob1=3/14
        "Probability Agent",
        # "2inpRxCnot1llr[e-2,e-2,e-2] (20)",
        "6inpRxCnot1llr[e-2,e-2,e-2] (24)",
        "6inpRxCnot2llr[e-2,e-2,e-2] (36)",
        "6inpRxCnot3llr[e-2,e-2,e-2] (48)",
        "6inpRxCnot4llr[e-2,e-2,e-2] (60)",
        "6inpRxCnot5llr[e-2,e-2,e-2] (72)",
        # "2inpNN1l2nlre-2 (24)",
        # "2inpNN1l2nlre-3 (24)",
        "6inpNN1l2nlre-2 (32)",
        # "6inpNN1l2nlre-3 (32)",
        # "2inpNN2l2nlre-2 (30)",
        # "2inpNN2l2nlre-3 (30)",
        # "6inpNN2l2nlre-2 (38)",
        "6inpNN2l2nlre-3 (38)",
        # "2inpNN3l10nlr0.001 (316)",
        # "1inpNN3l10nlr0.001 (306)",
        # "2inpNN3l10nlr0.001 (316)",
        # "3inpNN3l10nlr0.001 (326)",
        # "4inpNN3l10nlr0.001 (336)",
        # "5inpNN3l10nlr0.001 (346)",
        "6inpNN3l10nlr0.001 (356)",
        # "10inpNN3l10nlr0.001 (396)",

        # #QFIAHv2 5 holes
        # "Probability Agent",
        # "2inpQiboRxCnot1llre-2 (8)",
        # "2inpRxCnot1llr[e-2,e-2,e-2] (17)",
        # "5inpRxCnot1llr[e-2,e-2,e-2] (20)",
        # "2inpRxRyRzCz1lr[e-2,e-2,e-2] (37)",
        # "2inpRxCnot3llr[e-2,e-2,e-2] (31)",
        # "2inpNN1l2nlre-2 (21)",
        # # "2inpNN2l2nlre-3 (27)",
        # # "5inpNN2l2nlre-3 (33)",
        # "2inpNN2l2nlre-2 (27)",
        # "5inpNN2l2nlre-2 (33)",
        # # "NN2inp3l10nlre-2 (305)",
        # "NN5inp3l10nlre-3 (335)"


        # QFIAHv1 10 holes
        # "NN 2 nodes 2 layers lr 0.001"

        # ## QFIAHv1 5 holes prob1=3/14
        # "Probability Agent",
        # "QiboRxCnot1llre-2 (8)",
        # "RxCnot1llr[e-2,e-2,e-2] (17)",
        # # "RxCnot2llr[e-2,e-2,e-2] (24)"
        # "RxCnot3llr[e-2,e-2,e-2] (31)",
        # "RxRyRzCz1llr[e-2,e-2,e-2] (37)",
        # # "RxRyRzCz2llr[e-2,e-2,e-2] (54)",
        # "RxRyRzCz3llr[e-2,e-2,e-2] (71)",
        # "NN2inp1l2nlre-2 (21)",
        # "NN2inp2l2nlre-3 (27)",
        # "NN5inp2l2nlre-3 (33)",
        # "NN2inp3l10nlre-3 (305)",
        # "NN5inp3l10nlre-3 (335)"

        # ## QFIAHv1 5 holes prob1 = prob2
        # "Probability Agent",
        # "2inpQiboRxCnot1llre-2 (8)",
        # "2inpRxCnot1llr[e-2,e-2,e-2] (17)",
        # "2inpRxRyRzCz1llr[e-2,e-2,e-2] (37)",
        # "5inpRxRyRzCz1llr[e-2,e-2,e-2] (40)",
        # "NN2inp1l2nlre-2 (21)",
        # "NN2inp2l2nlre-3 (27)",
        # "NN2inp3l10nlre-3 (305)",
        # "NN5inp3l10nlre-3 (335)"

        # # FIAH 10 holes
        # "Probability agent",
        # # "RxCnot1llr[e-2,e-2,e-2] (32)",
        # # "RxCnot1llr[e-2,e-2,e-2] (32)",
        # "RxCnot1llr[e-2,e-3,e-2] (32)",
        # # "RxCnot1lr[e-2,e-3,e-2]",
        # # "RxRyRzCz1llr[e-2,e-3,e-2] (72)",
        # "NN1l2nlre-2 (36)",
        # # "NN1l2nlre-3 (36)",
        # "NN2l2nlre-2 (42)",
        # # "NN2l2nlre-3 (42)",
        
        # "RxCnot2llr[e-2,e-3,e-2] (44)", 
        # # "RxRyRzCz2llr[e-2,e-3,e-2] (104)",
        # "RxCnot3llr[e-2,e-3,e-2] (56)",
        # # "RxRyRzCz3llr[e-2,e-3,e-2] (136)",
        # # "RxCnot5llr[e-2,e-2,e-2] (80)",
        # "RxCnot5llr[e-2,e-3,e-2] (80)",
        # # "RxRyRzCz5llr[e-2,e-2,e-2] (200)",
        # # "RxRyRzCz5llr[e-2,e-3,e-2] (200)",
        # "RxRyRzCz5llr[e-2,e-3,e-2] (200)",
        
        # "NN3l6nlre-3 (172)",
        # # "NN3l10nlre-3 (360)",
        # # "NN3l10nlre-4 (360)",
        # # "RxCnot10llr[e-2,e-3,e-2] (140)",
        # "RxRyRzCz10llr[e-2,e-3,e-2] (360)",
        # # "RxCnot20llr[e-2,e-3,e-2] (260)",
        # # "RxRyRzCz20llr[e-2,e-3,e-2] (680)",


        # # Classical FIAH 6 holes
        # "Probability Agent",
        # "2inpRxCnot1llr[e-2,e-2,e-2] (20)",
        # "6inpRxCnot1llr[e-2,e-2,e-2] (24)",
        # "2inpNN1l2nlre-2 (24)",
        # "6inpNN1l2nlre-2 (32)",
        # "2inpNN2l2nlre-2 (30)",
        # "6inpNN2l2nlre-2 (38)",
        # "2inpNN3l10nlre-3 (316)",
        # "6inpNN3l10nlre-3 (356)"

        # # FIAH 5 holes
        # "Probability Agent",
        # # "NN 3 inp 3 layers 3 nodes (44 params)",
        # "NN3l10nlre-3 (305)",
        # # "NN 3 layers 10 nodes lr 0.0005 (305 params)",
        # "NN3l4nlre-3 (77)",
        # # "NN 3 layers 4 nodes lr 0.0005 (77 params)",
        # # "NN 3 layers 3 nodes (53 params)",
        # # "NN 3 layers 2 nodes (33 params)",
        # # "NN 2 layers 5 nodes (75 params)",
        # # "NN 2 layers 3 nodes (41 params)",
        # "NN2l2nlre-2 (27)",
        # # "NN 2 layers 2 nodes (27 params)",
        # "NN1l2nlre-2 (21)",
        # # "NN 1 layer 2 nodes (21 params)",
        # "RxCnot1llre-2 (17)",
        # # "PQC 1 layer RxCnot lr[0.1,0.01,0.1] (17 params)",
        # # "PQC 1 layer RxCnot lr[0.1,0.1,0.1] (17 params)",
        # # "PQC Qibo 3 qubits 1 layer RxRyRz Cz (20 params)",
        # "QiboRxCnotlre-2 (8)"
        # # "PQC 5 qubits 1 layer Rx Cnot no obs weights (12 params)"



        



    ]
    n_eps = 250000
    smoothing = 5000

    plot_length_from_path_best(paths, labels, smoothing, n_eps, n_best=2)
    plt.legend()
    plt.savefig("plot_best.pdf")
    plt.close()

    plot_length_from_path(paths, labels, smoothing, n_eps, alpha = 0.1, mean= True)
    plt.legend()
    plt.savefig("plot_mean.pdf")
    plt.close()

    