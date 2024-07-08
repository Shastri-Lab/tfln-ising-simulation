#%%
import scipy.io
import numpy as np
from collections import defaultdict
from dimod.utilities import qubo_to_ising

def flatten_spin_matrix(spin_matrix, idx_vector):
    spin_vector = []
    for ((x_idx, y_idx), f) in idx_vector:
        spin_vector.append(spin_matrix[f, y_idx, x_idx])
    return np.array(spin_vector)

def vector_to_spin_matrix(spin_vector, idx_vector):
    L = max(map(lambda x: x[1], idx_vector)) + 1
    M = max(map(lambda x: x[0][0], idx_vector)) + 1
    N = max(map(lambda x: x[0][1], idx_vector)) + 1
    spin_matrix = np.zeros((L, N, M))
    for spin, ((x_idx, y_idx), f) in zip(spin_vector, idx_vector):
        spin_matrix[f, y_idx, x_idx] = spin
    return spin_matrix

def J_dict_to_mat(J_dict, idx_vector):
    J_dict = defaultdict(float, J_dict)
    num_indices = len(idx_vector)
    J = np.zeros(shape=(num_indices, num_indices))
    for i, idx1 in enumerate(idx_vector):
        for j in range(i+1, num_indices):
            idx2 = idx_vector[j]
            J[i, j] = J_dict.get((idx1, idx2), J_dict.get((idx2, idx1), 0.0))
            J[j, i] = J[i, j]
    return J

def h_dict_to_mat(h_dict, idx_vector):
    h_dict = defaultdict(float, h_dict)
    num_indices = len(idx_vector)
    h = np.zeros(shape=(num_indices))
    for i, idx in enumerate(idx_vector):
        h[i] = h_dict[idx]
    return h

def save_qubo_model_to_ising_mat(hp_qubo_model, filename, target_energy=0.0):
    Q_qubo = hp_qubo_model.interaction_matrix()
    h_ising, J_ising, offset_ising = qubo_to_ising(Q_qubo)
    L = len(hp_qubo_model.sequence)
    N, M = hp_qubo_model.dim
    h = h_dict_to_mat(h_ising, hp_qubo_model.keys)
    J = J_dict_to_mat(J_ising, hp_qubo_model.keys)
    scipy.io.savemat(filename, {
        'h': h,
        'J': J,
        'offset': offset_ising,
        'L': L,
        'N': N,
        'M': M,
        'keys': hp_qubo_model.keys,
        'sequence': hp_qubo_model.sequence,
        'target_energy': target_energy,
    })


# %%
