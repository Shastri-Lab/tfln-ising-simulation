#%%
import os
from os import path
import scipy.io
import numpy as np
from collections import defaultdict
from dimod.utilities import qubo_to_ising

def get_matrix_density(J):
    return np.count_nonzero(J) / J.size

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
        for j, idx2 in enumerate(idx_vector):
            J[i, j] = J_dict[idx1, idx2]
    return J

def h_dict_to_mat(h_dict, idx_vector):
    h_dict = defaultdict(float, h_dict)
    num_indices = len(idx_vector)
    h = np.zeros(shape=(num_indices))
    for i, idx in enumerate(idx_vector):
        h[i] = h_dict[idx]
    return h

def save_model_for_matlab(model, filename):
    h_ising, J_ising, offset_ising = model.to_ising()
    L = len(model.sequence)
    N, M = model.dim
    h = h_dict_to_mat(h_ising, model.keys)
    J = J_dict_to_mat(J_ising, model.keys)
    scipy.io.savemat(filename, {
        'h': h, 'J': J,
        'L': L, 'N': N, 'M': M,
        'lambdas': model.Lambda, # TODO: this is only used for protein folding, should be removed for general case
        'keys': model.keys, # TODO: this is only used for protein folding, should be removed for general case
        'offset': offset_ising,
        'sequence': model.sequence, # TODO: this is only used for protein folding, should be removed for general case
        'target_energy': model.target_energy, # TODO: this is only used for protein folding, should be removed for general case
    })

def save_results(model, e_history, bits_history, x_vector, alpha_beta, noise_std):
    data_dir = path.join(path.dirname(path.abspath(__file__)), 'results')
    os.makedirs(data_dir, exist_ok=True) # create the directory if it doesn't exist
    results_filename = path.join(data_dir, f'{model.name}_{model.dim[1]}x{model.dim[0]}')
    i = 1
    filename = results_filename + '.npz'
    while path.exists(filename):
        filename = f'{results_filename}_{i}.npz'
        i += 1

    # take every 10th bit to reduce the size of the file
    # make sure last bits is included
    bits_history = np.array(bits_history)
    bits = bits_history[:, :, ::10]
    bits[:, :, -1] = bits_history[:, :, -1]

    # split alpha_beta into alphas and betas
    alphas = alpha_beta[:, 0]
    betas = alpha_beta[:, 1]

    print(f'Saving results to {filename}...', end=' ')
    np.savez(
        filename,
        e_history=e_history,
        bits_history=bits,
        x_vector=x_vector,
        alphas=alphas, 
        betas=betas,
        sequence=model.sequence, # TODO: this is only used for protein folding, should be removed for general case
        target_energy=model.target_energy, # TODO: this is only used for protein folding, should be removed for general case
        lambdas=model.Lambda, # TODO: this is only used for protein folding, should be removed for general case
        noise_std=noise_std,
        )
    print('Done.')
