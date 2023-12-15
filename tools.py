import numpy as np
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

if __name__ == "__main__":
    matrix = generate_givens_wall(6, "gx", 0.25*np.pi, "gy", 0.25*np.pi)
    print(matrix)

