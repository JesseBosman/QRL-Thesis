import numpy as np
from time import time
from argparse import ArgumentParser
from tools import retrieve_transfer_matrices


def main(n_guesses, n_holes, transfer_matrices, env_name, brick1, theta1, brick2, theta2):       

    starttime1 = time()

    if env_name.lower() == "fiah":
        initial_state = np.ones(n_holes)/n_holes
        run_guesses_func = run_guesses_fiah
    elif env_name.lower()== "givens":
        initial_state = np.ones(n_holes)/np.linalg.norm(np.ones(n_holes))
        run_guesses_func = run_guesses_givens
    
    elif env_name.lower() == "qfiah":
        raise ValueError("QFIAH disabled due to work in progress.")
    
    else:
        raise ValueError("Undefined env name.")

    min_performance = np.inf
    min_configuration = []

        
    for guesses in configurations(n_guesses, n_holes):
        # print(guesses)
        # starttime = time()
        avg = run_guesses_func(initial_state, transfer_matrices, guesses, n_guesses)
        
        if avg < min_performance:
            min_performance= avg
            min_configuration = guesses
        # runtime = time()-starttime
        # if i%1000 == 0:
        #     print(f"Strategy: {i}", end="\r")
        # print(f"Strategy: {i}, average number of guesses {avg:.2f}", end="\r")
        # avgs[str(guesses)] = avg
        # avgs[guesses] = avg
        # avgs[i] = avg
        # i += 1
    runtime1 = time()-starttime1

    # print(avgs)
    # print(f"{n_guesses}: {min(avgs.values()):.2f} {min(avgs, key=avgs.get)}, {runtime1:.4f}, {i}                                                                                                             ")
    # min_performance = np.inf
    # min_configuration = []
    # for i, configuration in enumerate(configurations(n_guesses, n_holes)):
    #     if avgs[i] < min_performance:
    #         min_performance = avgs[i]
    #         min_configuration = configuration
    print(f"Environment: {env_name} {n_holes} holes {brick1}{theta1}{brick2}{theta2}")
    print(f"{n_guesses}: {min_performance:.4f} {min_configuration}, {runtime1:.2f}, {i}                                                                                                             ")
    # print(min(avgs))


def configurations(N, n_holes):
    # print(n)
    if N == 1:
        for i in range(n_holes):
            yield [i]
    else:
        for configuration in configurations(N-1, n_holes):
            for i in range(n_holes):
                yield np.concatenate(([i], configuration))  


def run_guesses_fiah(initial_state, transfer_matrix, guesses, n_guesses):
    state = initial_state.copy()
    avg = 0
    for n, guess in enumerate(guesses):
        find_prob = state[guess]

        avg += find_prob*(n+1)
        state[guess] = 0
        state = np.inner(transfer_matrix,state)

        # state = state/np.sum(state) # NORMALISE (Not the correct calculation)
        # print(f"{n+1} {avg:.2f} {state} {np.sum(state)}")

    avg += (n_guesses+1)*np.sum(state)

    return avg  

def run_guesses_givens(initial_state, transfer_matrix, guesses, n_guesses):
    state = initial_state.copy()
    avg = 0
    for n, guess in enumerate(guesses):
        find_prob = np.abs(state[guess])**2

        avg += find_prob*(n+1)
        state[guess] = 0
        state = np.inner(transfer_matrix,state)

        # state = state/np.sum(state) # NORMALISE (Not the correct calculation)
        # print(f"{n+1} {avg:.2f} {state} {np.sum(state)}")

    avg += (n_guesses+1)*np.sum(np.square(np.abs(state)))

    return avg

def run_guesses_qfiah(initial_state, transfer_matrices, guesses, n_guesses):
    state = initial_state.copy()
    avg = 0
    for n, guess in enumerate(guesses):
        avg += state[guess]*(n+1)
        state[guess] = 0
        if n%2 == 0:
            state = np.inner(transfer_matrices[0],state)
        else:
            state = np.inner(transfer_matrices[1], state)

        # state = state/np.sum(state) # NORMALISE (Not the correct calculation)
        # print(f"{n+1} {avg:.2f} {state} {np.sum(state)}")
    avg += (n_guesses+1)*np.sum(state)
    return avg



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-N", "--n_guesses", default=10, type=int, help="Number of guesses")
    argparser.add_argument("-H", "--n_holes", default=5, type=int, help="Number of holes")
    argparser.add_argument("-E", "--env_name", default= "FIAH", type = str, help= "The environment")
    argparser.add_argument("--prob_1", "-p1", type = float, default= 3/14, help="Probability of split to the right for QFIAH.")
    argparser.add_argument("--tunneling_prob", "-tp", type = float, default = 0.2, help= "The tunneling probability for the QFIAH environment.")
    argparser.add_argument("--brick1", "-b1", type = str, default="gx", help="The type of brick for the 1st brick in the Givens wall. Note that due ot rules of matrix multiplication, the 2nd brick comes first in the wall.")
    argparser.add_argument("--theta1", "-t1", type = float, default=0.25, help="The rotation parameter for the 1st brick in the Givens wall. Will be mutliplied with pi.")
    argparser.add_argument("--brick2", "-b2", type = str, default="gy", help="The type of brick for the 2nd brick in the Givens wall.")
    argparser.add_argument("--theta2", "-t2", type = float, default=0.25, help="The rotation parameter for the 2nd brick in the Givens wall. Will be muliplied with pi.")

    args = argparser.parse_args()
    N = args.n_guesses
    n_holes = args.n_holes
    env_name = args.env_name
    prob_1 = args.prob_1
    tunneling_prob = args.tunneling_prob
    brick1= args.brick1
    theta1= args.theta1 *np.pi
    brick2 = args.brick2
    theta2 = args.theta2 * np.pi

    print("Parameters are: n_guesses {}, n_holes {}, env {}, b1 {}, t1 {}, b2 {}, t2 {}".format(
        N, n_holes, env_name, brick1, theta1, brick2, theta2))

    transfer_matrices = retrieve_transfer_matrices(env_name, n_holes, prob_1, tunneling_prob, brick1, theta1, brick2, theta2)


    for i in range(1, N+1):
        main(i, n_holes, transfer_matrices, env_name, brick1, theta1, brick2, theta2)
        
