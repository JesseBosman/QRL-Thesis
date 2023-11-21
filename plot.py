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


def plot_length_from_path_best(path_folders, labels, smoothing, n_eps):
    """
    This functions will plot the average episode reward over a ~ thousand budget step window. It takes as input a folder directory containing
    .npy files with the data. Averages will be taken across these files to create statistically meaningfull results.
    """
    # os.listdir
    X = np.linspace(0, n_eps, int(n_eps/smoothing), endpoint= False)

    for p, path_folder in enumerate(path_folders):
        n_files = 0
        for path in os.listdir(path_folder):
            n_files += 1

        best_final_value = np.inf
        best_run= None

        for i, path in enumerate(os.listdir(path_folder)):
            dir = os.path.join(path_folder, path)
            episode_lengths = np.load(
                dir)[:n_eps]  # now names episode_rewards, but depending on dir can also be episode_lengths
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

            final_value = smoothed_episode_lenghts[-1]
            if final_value < best_final_value:
                best_final_value = final_value
                best_run = smoothed_episode_lenghts
        
        plt.plot(X, best_run, label=labels[p])
        print("The amount of files in path {} was {}".format(p, n_files))

    plt.ylabel("Episode lengths")
    plt.xlabel("Episode")

    pass


# adjust to own paths
if __name__ == '__main__':
    paths = [
        # # QFIAHv2 5 holes
        
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes2layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes3layers10nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv2-prob10.21-prob20.79-maxsteps65holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/"
        
        # # QFIAHv1 5 holes prob1!=prob2 (done)
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits2layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-prob10.21-prob20.79-maxsteps6-PQC-v45holes5qubits3layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "lengthsQuantumProbabilityAgent5holes1000000neps6steps0.21428571428571427prob10.7857142857142857prob2/",
        # # "lengthsQuantumProbabilityAgent5holes1000000nepsinfsteps0.21428571428571427prob10.7857142857142857prob2/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21-prob20.79-maxsteps65holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv1-prob10.21428571428571427-prob_20.7857142857142857-maxsteps65holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"

        # # QFIAHv1 5 holes prob1=prob2 (done)
        # "lengthsQuantumProbabilityAgent5holes1000000neps6steps/",
        # # "lengthsQuantumProbabilityAgent5holes1000000nepsinfsteps/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-PQC-v4-RXCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-QFIAHv1-PQC-v45holes5qubits1layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-QFIAHv1-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv1-prob10.5-prob20.5-maxsteps65holes1layers2nodeslr0.01neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv15holes2layers2nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-QFIAHv15holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/5inp-QFIAHv15holes3layers10nodeslr0.001neps250000bsize10gamma1start1anil0.25elu/"

        # # Classical FIAH 10 holes (need to compare 3 layer PQC with similar NN)
        # # "lengthsProbabilityAgent10holes1000000neps/",
        # # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits5layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/"
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes1layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits1layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits1layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes1layers2nodeslr0.01neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes1layers2nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes2layers2nodeslr0.01neps500000bsize10gamma1start1anil0.25elu/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes2layers2nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits2layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits2layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits3layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v410holes10qubits3layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v4-RxCNOT10holes10qubits3layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits5layersneps1000000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers6nodeslr0.001neps1000000bsize10gamma1start1anil0.25elu/"
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes5layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits5layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits5layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT10holes10qubits5layersneps500000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits10layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers6nodeslr0.001neps1000000bsize10gamma1start1anil0.25elu/"
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers10nodeslr0.001neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/NN/ep_length/2inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps1610holes3layers10nodeslr0.0001neps500000bsize10gamma1start1anil0.25elu/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps16-PQC-v410holes10qubits20layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/"
        
        # # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v410holes10qubits10layersneps500000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/"
       
       
        # # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes5qubits1layersneps250000lrin0.1lrvar0.1lrout0.1bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes5qubits1layersneps250000lrin0.1lrvar0.01lrout0.1bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        

        ## Classical FIAH 5 holes (done)
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
        
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/5-inp-PQC-v35holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/3-inp-PQC-v35holes1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v45holes3qubits1layersneps250000lrin0.01lrvar0.001lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.1lrvar0.01lrout0.1bsize10gamma1start1anil0.25/",
        "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT5holes5qubits1layersneps250000lrin0.1lrvar0.1lrout0.1bsize10gamma1start1anil0.25/"
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-maxsteps6-red_PQC-RxCNOT5holes3qubits1layersneps250000lr0.01bsize1gamma1start1anil0.25/"
        # "/home/s2025396/data1/resultsQRL/PQC/ep_length/2-inp-FoxInAHolev2-prob10.21-prob20.79-maxsteps6-PQC-v4-RxCNOT-noobsweights5holes5qubits1layersneps250000lrin0.01lrvar0.01lrout0.01bsize10gamma1start1anil0.25/"

    ]

       
    labels = [
        # #QFIAHv2 5 holes
        # "2 inp PQC RxCnot 1 layer (17 params)",
        # "5 inp PQC RxCnot 1 layer (20 params)",
        # "NN 2 inp 1 layer 2 nodes lr 0.01 (21 params)",
        # # "PQC RxRyRz 1 layer (37 params)",
        # # "NN 2inp 2 layers 2 nodes lr 0.001 (27 params)",
        # # "NN 5inp 2 layers 2 nodes lr 0.001 (33 params)",
        # "NN 2inp 2 layers 2 nodes lr 0.01 (27 params)",
        # "NN 5inp 2 layers 2 nodes lr 0.01 (33 params)",
        # "NN 2 inp 3 layers 3 nodes lr 0.01 (326 params)",
        # "NN 5 inp 3 layers 3 nodes lr 0.001 (335 params)"


        # QFIAHv1 10 holes
        # "NN 2 nodes 2 layers lr 0.001"

        # ## QFIAHv1 5 holes prob1=3/14
        # "PQC 1 layer RxCnot (17 params)",
        # "PQC 3 layers RxCnot (31 params)",
        # "PQC 1 layer RxRyRz Cz (37 params)",
        # "PQC 3 layers RxRyRz Cz (71 params)",
        # "NN 2 inp 1 layer 2 nodes lr e-2 (21 params)",
        # "NN 2 inp 2 layers 2 nodes (27 params)",
        # "NN 5 inp 2 layers 2 nodes (33 params)",
        # "NN 2 inp 3 layers 10 nodes (305 params)",
        # "NN 5 inp 3 layers 10 nodes (335 params)"

        # # ## QFIAH 5 holes prob1 = prob2
        # "Probability Agent",
        # "PQC 2 inp 1 layer RxCnot (17 params)",
        # "PQC 2 inp 1 layer RxRyRzCz (37 params)",
        # "PQC 5 inp 1 layer RxRyRzCz (40 params)",
        # "NN 2 inp 1 layer 2 nodes lr e-2 (21 params)",
        # "NN 2 inp 2 layers 2 nodes (27 params)",
        # "NN 2 inp 3 layers 10 nodes (305 params)",
        # "NN 5 inp 3 layers 10 nodes (335 params)"

        # # FIAH 10 holes
        # # "Probability agent",
        # # "PQC RxRyRz CZ 5 layers"
        # # "RxRyRz Cz 1 layer",
        # # # "RxRyRz Cz lr [0.01, 0.001, 0.01]",
        # "Rx Cnot 1 layer (32 params)",
        # "NN 1 layer 2 nodes lr e-2 (36 params)",
        # "NN 1 layer 2 nodes lr e-3 (36 params)",
        # "NN 2 layers 2 nodes lr e-2 (42 params)",
        # "NN 2 layers 2 nodes lr e-3 (42 params)",
        # # "Rx Cnot 1 layer lr [0.01, 0.01, 0.01]"
        
        # # "RxRyRz Cz 2 layers",
        # # "Rx Cnot 2 layers",
        # # "RxRyRz Cz 1 layer lr [0.01, 0.001, 0.01]",
        # # "Rx Cnot 1 layer lr [0.01, 0.001, 0.01]",
        # # "Rx Cnot 1 layer lr [0.01, 0.01, 0.01]",
        # # "RxRyRz Cz 3 layers",
        # # "Rx Cnot 3 layers",
        # # "RxRyRz Cz 3 layer [0.01,0.001.0.01] (136 params)",
        # "RxRyRz Cz 5 layers (200 params)",
        # "NN 3 layer 6 nodes (172 params)"
        # # "RxRyRz Cz 5 layers lr [0.01, 0.01, 0.01]",
        # # "Rx Cnot 5 layers",
        # # "Rx Cnot 5 layers lr [0.01, 0.01, 0.01]",
        # # "RxRyRz Cz 3 layer lr [0.01, 0.001, 0.01]"
        # # "RxRyRz Cz 10 layers",
        # # "NN 3 layer 6 nodes lr 0.001 (172 params)"
        # # "NN 3 layers 10 nodes lr 0.001",
        # # "NN 3 layers 10 nodes lr 0.0001",
        # # "RxRyRz Cz 20 layers"

        # FIAH 5 holes
        # # "NN 3 inp 3 layers 3 nodes (44 params)",
        # "NN 3 layers 10 nodes (305 params)",
        # # "NN 3 layers 10 nodes lr 0.0005 (305 params)",
        # "NN 3 layers 4 nodes (77 params)",
        # # "NN 3 layers 4 nodes lr 0.0005 (77 params)",
        # # "NN 3 layers 3 nodes (53 params)",
        # # "NN 3 layers 2 nodes (33 params)",
        # # "NN 2 layers 5 nodes (75 params)",
        # # "NN 2 layers 3 nodes (41 params)",
        # "NN 2 layers 2 nodes lr e-2 (27 params)",
        # # "NN 2 layers 2 nodes (27 params)",
        # "NN 1 layer 2 nodes lr e-2 (21 params)",
        # "NN 1 layer 2 nodes (21 params)",
        "PQC 5 qubits 1 layer RxCnot lr[0.01,0.01,0.01] (17 params)",
        "PQC 5 qubits 1 layer RxCnot lr[0.1,0.01,0.1] (17 params)",
        "PQC 5 qubits 1 layer RxCnot lr[0.1,0.1,0.1] (17 params)"
        # "PQC Qibo 3 qubits 1 layer RxRyRz Cz (20 params)"
        # "PQC 5 qubits 1 layer Rx Cnot no obs weights (12 params)"



        



    ]
    n_eps = 250000
    smoothing = 5000

    plot_length_from_path(paths, labels, smoothing, n_eps, alpha = 0.1, mean= True)
    # plot_reward_from_path(paths, labels, 5000)
    plt.legend()
    plt.savefig("plot_mean.pdf")
    plt.savefig("plot_mean.png", format = "png", dpi = 1200)
    plt.close()

    plot_length_from_path_best(paths, labels, smoothing, n_eps )
    plt.legend()
    plt.savefig("plot_best.pdf")
    plt.savefig("plot_best.png", format = "png", dpi = 1200)
    plt.close()