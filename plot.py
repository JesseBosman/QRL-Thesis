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
        "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",  
        "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/"
        # "lengthsQuantumProbabilityAgent5holes1000000neps6steps0.21428571428571427prob10.7857142857142857prob2/",
        # "lengthsQuantumProbabilityAgent5holes1000000nepsinfsteps0.21428571428571427prob10.7857142857142857prob2/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"


        # "lengthsQuantumProbabilityAgent5holes1000000neps6steps/",
        # "lengthsQuantumProbabilityAgent5holes1000000nepsinfsteps/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-PQC-v4-RXCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-PQC-v45holes5qubits1layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-QFIAHv1-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv15holes2layers2node250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv15holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv15holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"

        # "lengthsProbabilityAgent10holes1000000neps/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits5layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/"
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes1layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits1layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits1layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits2layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits3layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes5layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits5layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits5layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits5layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits10layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers10nodeslr0.0001neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v410holes10qubits20layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/"
        
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits10layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/"
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes5qubits1layersneps250000lrin0.1lrvar0.1lrout0.1bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes5qubits1layersneps250000lrin0.1lrvar0.01lrout0.1bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        


        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/3inp-FoxInAHolev25holes2layers3nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers10nodeslr0.0005neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers4nodeslr0.0005neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers4nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes3layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes2layers5nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes2layers3nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev25holes1layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-PQC-v35holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/3-inp-PQC-v35holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes3qubits1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/"
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT-noobsweights5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/"

    ]

       
    labels = [

        # ## QFIAH 5 holes
        "2 inp NN 2 layers 2 nodes (27 params)",
        # "5 inp NN 2 layers 2 nodes",
        "2 inp NN 3 layers 10 nodes (305 params)",
        # "5 inp NN 3 layers 10 nodes",
        "2 inp PQC 1 layer RxCnot (17 params)",
        # "2 inp PQC 2 layers RxCnot (24 params)",
        "2 inp PQC 3 layers RxCnot (31 params)",
        "2 inp PQC 1 layer RxRyRz CZ (37 params)",
        # "2 inp PQC 2 layer RxRyRz CZ (54 params)",
        "2 inp PQC 3 layer RxRyRz CZ (71 params)"

        # "Probability agent 6 steps",
        # "Probability agent inf steps",
        # "2 inp 2 layers 2 nodes",
        # "5 inp 2 layers 2 nodes",
        # "2 inp 3 layers 10 nodes",
        # "5 inp 3 layers 10 nodes"
        # "Quantum Probability agent 5 holes 6 steps",
        # "Quantum Probability agent 5 holes unbounded",
        # "2 inp PQC Rx Cnot 1 layer",
        # "2 inp PQC RxRyRz Cz 1 layer",
        # "5 inp PQC RxRyRz Cz 1 layer",
        # "2 inp NN 2 layers 2 nodes",
        # "2 inp NN 3 layers 10 nodes",
        # "5 inp NN 3 layers 10 nodes"

        # FIAH 10 holes
        # "Probability agent",
        # "PQC RxRyRz CZ 5 layers"
        # "RxRyRz Cz 1 layer lr [0.01, 0.001, 0.01]",
        # # "RxRyRz Cz lr [0.01, 0.001, 0.01]",
        # "Rx Cnot 1 layer lr [0.01, 0.001, 0.01]",
        # "Rx Cnot 1 layer lr [0.01, 0.01, 0.01]"

        # "RxRyRz Cz 1 layer lr [0.01, 0.001, 0.01]",
        # "Rx Cnot 1 layer lr [0.01, 0.001, 0.01]",
        # "Rx Cnot 1 layer lr [0.01, 0.01, 0.01]",
        # "RxRyRz Cz 3 layers lr [0.01, 0.001, 0.01]",
        # "RxRyRz Cz 5 layers lr [0.01, 0.001, 0.01]",
        # "RxRyRz Cz 5 layers lr [0.01, 0.01, 0.01]",
        # "Rx Cnot 5 layers lr [0.01, 0.001, 0.01]",
        # "Rx Cnot 5 layers lr [0.01, 0.01, 0.01]",
        # "RxRyRz Cz 10 layers lr [0.01, 0.001, 0.01]",
        # "NN 3 layers 10 nodes lr 0.001",
        # "NN 3 layers 10 nodes lr 0.0001",
        # "RxRyRz Cz 20 layers lr [0.01, 0.001, 0.01]"

        ## FIAH 5 holes
        # # "NN 3 inp 3 layers 3 nodes (44 params)",
        # "NN 3 layers 10 nodes (305 params)",
        # # "NN 3 layers 10 nodes lr 0.0005 (305 params)",
        # "NN 3 layers 4 nodes (77 params)",
        # # "NN 3 layers 4 nodes lr 0.0005 (77 params)",
        # # "NN 3 layers 3 nodes (53 params)",
        # # "NN 3 layers 2 nodes (33 params)",
        # "NN 2 layers 5 nodes (75 params)",
        # # "NN 2 layers 3 nodes (41 params)",
        # "NN 2 layers 2 nodes (27 params)",
        # "NN 1 layer 2 nodes (21 params)",
        # "PQC 5 qubits 1 layer Rx Cnot(17 params)",
        # "PQC 5 qubits 1 layer Rx Cnot no obs weights (12 params)"



        



    ]
    

    plot_length_from_path(paths, labels, 5000, 500000, alpha = 0.1, mean= True)
    # plot_reward_from_path(paths, labels, 5000)
    plt.legend()
    plt.savefig("plot.pdf")
    plt.savefig("plot.png", format = "png", dpi = 1200)