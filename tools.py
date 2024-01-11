import numpy as np
import csv
from time import time
import os
import pickle
from tqdm import tqdm
# import tensorflow_quantum as tfq
def givens_x(theta, n_qubits, qubit_nr1, qubit_nr2):
    matrix = np.eye(n_qubits, dtype= np.complex128)
    base1 = min(qubit_nr1, qubit_nr2)
    base2 = max(qubit_nr1, qubit_nr2)
    matrix[base1, base1]= np.cos(theta)
    matrix[base1, base2] = -1j*np.sin(theta)
    matrix[base2, base1] = -1j*np.sin(theta)
    matrix[base2, base2]= np.cos(theta)
    return matrix

def givens_y(theta, n_qubits, qubit_nr1, qubit_nr2):
    matrix = np.eye(n_qubits, dtype= np.complex128)
    base1 = min(qubit_nr1, qubit_nr2)
    base2 = max(qubit_nr1, qubit_nr2)
    matrix[base1, base1]= np.cos(theta)
    matrix[base1, base2] = -np.sin(theta)
    matrix[base2, base1] = np.sin(theta)
    matrix[base2, base2]= np.cos(theta)
    return matrix

def givens_z(theta, n_qubits, qubit_nr1, qubit_nr2):
    matrix = np.eye(n_qubits, dtype= np.complex128)
    base1 = min(qubit_nr1, qubit_nr2)
    base2 = max(qubit_nr1, qubit_nr2)
    matrix[base1, base1]= np.exp(-1j*theta)
    matrix[base2, base2]= np.exp(1j*theta)
    return matrix

def generate_givens_wall(n_qubits, brick1, theta1, brick2, theta2):
    if brick1=="gx":
        function1 = givens_x
    elif brick1=="gy":
        function1= givens_y
    else:
        function1= givens_z

    if brick2=="gx":
        function2 = givens_x
    elif brick2=="gy":
        function2= givens_y
    else:
        function2= givens_z

    working_matrix = np.eye(n_qubits)
    for i in range(0, n_qubits-1, 2):
        working_matrix = np.matmul(function1(theta1, n_qubits, i, i+1), working_matrix)

    for j in reversed(range(1,n_qubits-1, 2)):
        working_matrix = np.matmul(function2(theta2, n_qubits, j, j+1), working_matrix)

    return working_matrix

def generate_qfiah_transfer_matrices(n_holes, prob_1, tunneling_prob):
    prob_2 = 1-prob_1
    split_1 = (prob_1*(1-tunneling_prob))
    split_2 = (prob_2*(1-tunneling_prob))
    tunnel_split_1 = (prob_1*tunneling_prob)
    tunnel_split_2 = (prob_2*tunneling_prob)

    T_odd = np.zeros(shape= (n_holes, n_holes))
    T_even = np.zeros(shape= (n_holes, n_holes))

    for i in range(n_holes -2):
        T_odd[i,i+1]= split_1
        T_odd[i+2, i+1]= split_2
        T_even[i, i+1]= split_2
        T_even[i+2, i+1]= -1*split_1

        try:
            T_odd[i,i+2]= tunnel_split_1
        except:
            pass
        try:
            T_odd[i+2, i]= tunnel_split_2
        except:
            pass

        try:
            T_even[i, i+2]= tunnel_split_2
        except:
            pass

        try:
            T_even[i+2, i]= -1*tunnel_split_1
        
        except:
            pass
    
    T_odd[-1,-2] = prob_2
    T_odd[0,1] = prob_1

    T_even[-1,-2] = -prob_1
    T_even[0,1] = prob_2
    
    # Hier nog tunneling aanpassen!

    T_odd[1,0]=(1-tunneling_prob)
    T_odd[2,0]= ((1-tunneling_prob)*tunneling_prob)
    T_odd[-2,-1]= (1-tunneling_prob)
    T_odd[-3,-1]= ((1-tunneling_prob)*tunneling_prob)
    T_even[1,0]=-(1-tunneling_prob)
    T_even[2,0]= -((1-tunneling_prob)*tunneling_prob)
    T_even[-2,-1]= (1-tunneling_prob)
    T_even[-3,-1]= ((1-tunneling_prob)*tunneling_prob)

    T_odd[-1,0]= tunneling_prob
    T_odd[0,-1]= tunneling_prob

    T_even[-1,0]= tunneling_prob
    T_even[0,-1]= -tunneling_prob

   
    return [T_even, T_odd]

def generate_fiah_transfer_matrix(n_holes):
    matrix = np.zeros((n_holes, n_holes))
    for i in range(1, n_holes-1):
        matrix[i-1, i]= 0.5
        matrix[i+1, i]= 0.5

    matrix[1,0]= 1
    matrix[-2,-1]=1
    return matrix

def retrieve_transfer_matrices(env_name, n_holes, prob_1, tunneling_prob, brick1, theta1, brick2, theta2):
    if env_name.lower() == "givens":
        return generate_givens_wall(n_holes, brick1, theta1, brick2, theta2)
    
    elif env_name.lower()== "qfiah":
        return generate_qfiah_transfer_matrices(n_holes, prob_1, tunneling_prob)
    
    elif env_name.lower()== "fiah":
        return generate_fiah_transfer_matrix(n_holes)
    
    else:
        raise ValueError("wrong environment name.")
    
def number_of_parameters_NN(len_state, n_nodes_per_layer, n_hidden_layers, n_holes):   
    """
    Calculates the number of parameters in a NN.
    """  
    return ((len_state + 1) * n_nodes_per_layer
        + (n_hidden_layers - 1) * (n_nodes_per_layer + 1) * n_nodes_per_layer
        + (n_nodes_per_layer + 1) * n_holes)

def number_of_parameters_PQC(len_state, n_layers, n_holes, RxCnot):   
    """
    Calculates the number of parameters in a PQC.
    """  
    if RxCnot:
        return (n_layers * (len_state + n_holes) + 2 * n_holes)
    else:
        return (n_layers * (len_state + 3 * n_holes) + 4 * n_holes)

def compare_amount_of_params(len_state, n_holes, RxCnot):
    print("NN's")
    for number_of_hidden_layers in range(1,4):
        print(f"amount of layers is {number_of_hidden_layers}")
        print(number_of_parameters_NN(len_state, 2, number_of_hidden_layers , n_holes))
        print(number_of_parameters_NN(len_state, 3, number_of_hidden_layers , n_holes))
        print(number_of_parameters_NN(len_state, 4, number_of_hidden_layers , n_holes))
        print(number_of_parameters_NN(len_state, 5, number_of_hidden_layers, n_holes))

    print("PQC's")
    for n in range(1,11):
        print(number_of_parameters_PQC(len_state, n, n_holes, RxCnot))

def write_results_to_pickle(
        results_dict, 
        len_state,
        n_holes,
        type, 
        specific_env_id, 
        n_nodes_per_layer=None, 
        n_hidden_layers=None, 
        n_layers=None, 
        RxCnot=None, 
        n_episodes=None, 
        learning_rate=None, 
        max_steps=None, 
        batch_size=None, 
        n_reps=None
        ):
    
    n_completed_runs = len(results_dict["avg_performance"])

    if n_completed_runs == 0:
        best_performance = np.NaN
        std_best_performance = np.NaN
        best_policy = [np.nan for _ in range(max_steps)]
        best_policy_math_avg = np.NaN
        n_achieved_best_policy = np.NaN
        mean_performance = np.NaN
        std_mean_performance = np.NaN
        std_std_performance = np.NaN
        
    else:
        best_performance_index = np.argmin(results_dict["policy_avg"])
        best_performance = results_dict["avg_performance"][best_performance_index]
        std_best_performance = results_dict["std"][best_performance_index]
        best_policy = results_dict["policy"][best_performance_index]
        best_policy_math_avg = results_dict["policy_avg"][best_performance_index]
        n_achieved_best_policy = np.sum(results_dict["policy_avg"] == best_policy_math_avg)
        mean_performance = np.mean(results_dict["avg_performance"])
        std_mean_performance = np.std(results_dict["avg_performance"])
        std_std_performance = np.sqrt(
            np.sum(np.square(results_dict["std"])) / (n_completed_runs**2)
        )  # std dev of the mean
    
    if type == "NN":
        number_of_parameters = number_of_parameters_NN(len_state, n_nodes_per_layer, n_hidden_layers, n_holes)
        data = {
            "environment": specific_env_id,
            "type": type,
            "n_inputs": len_state,
            "n_episodes": n_episodes,
            "n_holes": n_holes,
            "n_hidden_layers": n_hidden_layers,
            "n_nodes_per_layer": n_nodes_per_layer,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "number_of_parameters": number_of_parameters,
            "n_reps": n_reps,
            "n_completed_runs": n_completed_runs,
            "best_performance": best_performance,
            "std_best_performance": std_best_performance,
            "mean_performance": mean_performance,
            "std_mean_performance": std_mean_performance,
            "std_std_performance": std_std_performance,
            "best_policy": best_policy,
            "best_policy_math_avg": best_policy_math_avg,
            "n_achieved_best_policy": n_achieved_best_policy,
        }
    
    elif type == "PQC":
        number_of_parameters = number_of_parameters_PQC(len_state, n_layers, n_holes, RxCnot)
        data = {
            "environment": specific_env_id,
            "type": type,
            "n_inputs": len_state,
            "n_episodes": n_episodes,
            "n_holes": n_holes,
            "n_layers": n_layers,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "number_of_parameters": number_of_parameters,
            "n_reps": n_reps,
            "n_completed_runs": n_completed_runs,
            "best_performance": best_performance,
            "std_best_performance": std_best_performance,
            "mean_performance": mean_performance,
            "std_mean_performance": std_mean_performance,
            "std_std_performance": std_std_performance,
            "best_policy": best_policy,
            "best_policy_math_avg": best_policy_math_avg,
            "n_achieved_best_policy": n_achieved_best_policy,
            "RxCnot": RxCnot,
        }

    else:
        raise ValueError

    directory = f"/home/s2025396/data1/resultsQRL/NEW/pickled_dicts/{specific_env_id}{n_holes}holes{time()}.pkl"
    with open(directory, "wb") as f:
        pickle.dump(data, f)
        f.close()

    pass

        

def write_results_to_csv():
    directory = f"/home/s2025396/data1/resultsQRL/NEW/pickled_dicts/"
    for path in tqdm(os.listdir(directory)):
        file_name = os.path.join(directory, path)
        with open(file_name,'rb') as f:
            data = pickle.load(f)
            f.close()

        os.remove(file_name)
        type = data['type']
        keys = list(data.keys())
        csv_path = f"/home/s2025396/data1/resultsQRL/NEW/"+type+"_experiment_data.csv"
        with open(csv_path, 'a') as f_object:
            dictwriter_object = csv.DictWriter(f_object, fieldnames=keys)

            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(data)

            # Close the file object
            f_object.close()
        
        pass





if __name__ == "__main__":
    write_results_to_csv()
    

