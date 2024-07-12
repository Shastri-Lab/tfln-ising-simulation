import os
import json
import numba
import numpy as np
from tqdm import tqdm
import os.path as path
from math import ceil, floor
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from dataclasses import dataclass
from hp_lattice import Lattice_HP_QUBO
from dwave.samplers import SimulatedAnnealingSampler
from dwave_to_isingmachine import (
    flatten_spin_matrix,
    vector_to_spin_matrix,
    J_dict_to_mat,
    h_dict_to_mat,
    save_model_for_matlab,
)

ROOT2 = np.sqrt(2)
def sigma(x): # return np.tanh(ROOT2*x)
    return sin(pi/2 * x) # equiv to: -1 + 2*np.cos(pi/4 * (x-1))**2
    
def load_hp_model_by_name(name, latdim=(10,10), lambdas=(2.1, 2.4, 3.0)):
    with open(path.join(path.dirname(path.abspath(__file__)), 'protein_sequences.json'), 'r') as f:
        hp_sequences = json.load(f)
    seq = next(filter(lambda x: x['name'] == name, hp_sequences)) # should only be one
    model = Lattice_HP_QUBO(
        name = name,
        dim = latdim,
        sequence = seq['sequence'],
        Lambda = lambdas,
        target_energy=seq['min_energy'],
    )
    return model

def plot_hp_convergence(e_history, qubo_bits, betas, target_energy=None):
    # e_history has shape (T, B, I)
    num_ics = e_history.shape[2]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title('Ising Energy')
    if target_energy is not None:
        ax[0].axhline(target_energy, color='k', linestyle='--')
    
    iters = list(range(e_history.shape[0]))
    for i, beta in enumerate(betas):
        if num_ics > 1: 
            ax[0].fill_between(iters, e_history[:, i, :].min(axis=-1), e_history[:, i, :].max(axis=-1), alpha=0.1, color=f'C{i}')
        # figure out which one has minimum energy at the last iteration
        min_energy_beta_idx = np.argmin(e_history[-1, i, :])
        ax[0].plot(iters, e_history[:, i, min_energy_beta_idx], '.-', lw=0.75, ms=2.0, label=f'β = {beta:.1e}', color=f'C{i}')
    
    ax[0].legend()
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Energy')
    
    final_energies = e_history[-1, :, :]
    # find index of the minimum energy
    min_energy_idx_flat = np.argmin(final_energies)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, final_energies.shape)
    min_energy = final_energies[min_energy_idx]
    min_qubo_bits = qubo_bits[*min_energy_idx, :]
    min_energy_beta = betas[min_energy_idx[0]]
    
    ax[1].set_title(f'Final Lattice Configuration (β = {min_energy_beta:.1e}, E = {min_energy:.1f})')
    model.show_lattice(min_qubo_bits, axes=ax[1])
    plt.show()

def save_mat_file(model):
    data_dir = path.join(path.dirname(path.abspath(__file__)), 'mat_files')
    os.makedirs(data_dir, exist_ok=True) # create the directory if it doesn't exist
    mat_filename = path.join(data_dir, f'{model.name}_{model.dim[1]}x{model.dim[0]}.mat')
    try:
        with open(mat_filename, 'r') as f:
            pass
    except FileNotFoundError:
        print(f'Saving model to {mat_filename}...', end=' ')
        save_model_for_matlab(model, mat_filename)
        print('Done.')

def print_final_energies(final_energies, betas, target_energy):
    print(f'Target Energy = {target_energy:.1f}\nFinal Energies:')
    for i, beta in enumerate(betas):
        energy = final_energies[i]
        energy = [round(e, 1) for e in energy]
        print(f'β = {beta:.1e}: E = {sum(energy):.1f} {energy}')

def save_results(model, e_history, bits_history, x_vector, betas, noise_std, asym):
    data_dir = path.join(path.dirname(path.abspath(__file__)), 'results')
    os.makedirs(data_dir, exist_ok=True) # create the directory if it doesn't exist
    results_filename = path.join(data_dir, f'{model.name}_{model.dim[1]}x{model.dim[0]}')
    if asym:
        results_filename += '_asym'
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

    print(f'Saving results to {filename}...', end=' ')
    np.savez(
        filename,
        e_history=e_history,
        bits_history=bits,
        x_vector=x_vector,
        betas=betas,
        sequence=model.sequence,
        target_energy=model.target_energy,
        lambdas=model.Lambda,
        noise_std=noise_std,
        )
    print('Done.')


def solve_hp_isingmachine(model, num_iterations=250_000, num_ics=2, betas=0.005, noise_std=0.125, asymmetric_J=True, is_plotting=True, is_saving=True):
    print(f'\nSetting up {model.name} simulation on {model.dim[1]}x{model.dim[0]} lattice...')
    target_energy = model.target_energy
    h_dict, J_dict, ising_e_offset = model.to_ising()
    h = h_dict_to_mat(h_dict, model.keys)
    J = J_dict_to_mat(J_dict, model.keys, asymmetric=asymmetric_J)
    save_mat_file(model)

    og_betas = np.atleast_1d(betas)
    betas = og_betas / np.max(np.abs(J))  # normalize beta by the maximum coupling strength
    alphas = 1 - betas                    # alpha is complement of beta for running average
    
    num_spins = h.shape[0]
    num_betas = betas.shape[0]

    W = np.zeros((num_betas, num_spins, num_spins))
    b = np.zeros((num_betas, num_spins))
    for i, (alpha, beta) in enumerate(zip(alphas, betas)):
        W[i, :, :] = alpha * np.eye(num_spins) - beta * J
        b[i, :] = -beta * h
    b = np.stack([b for _ in range(num_ics)], axis=1)

    x_init = np.random.uniform(-1, 1, (num_ics, num_spins))
    x_vector = np.stack([x_init for _ in range(num_betas)]) # use the same initial state for all betas
    noise = np.random.normal(0, noise_std, (num_ics, num_spins, num_iterations))
    noise = np.stack([noise for _ in range(num_betas)])
    output = np.zeros_like(x_vector)

    bits_history = []
    e_history = []

    print('Running simulation...')
    try:
        for t in tqdm(range(num_iterations)):
            spin_vector = np.sign(x_vector)     # σ ∈ {-1, 1}
            qubo_bits = (spin_vector+1)/2       # q ∈ {0, 1}
            
            # compute the energy of the current state
            current_energy = np.einsum('ijk,ijk->ij', spin_vector, np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h, spin_vector) + ising_e_offset + model.Lambda[0] * model.len_of_seq

            # record the history
            bits_history.append(qubo_bits.astype(bool))
            e_history.append(current_energy)
            
            # break if we are close enough to the target energy
            if np.any(np.abs(current_energy - target_energy) < 1e-3):
                break

            # compute the next state of the system
            np.einsum('ijk,ihk->ihj', W, sigma(x_vector), out=output) # W has shape (B, N, N); x_vector has shape (B, I, N); the output has shape (B, I, N)
            output += b + noise[:,:,:,t]
            x_vector = output
            # x_vector -= np.mean(x_vector, axis=1, keepdims=True) # subtract the mean
            x_vector /= np.max(np.abs(x_vector), axis=1, keepdims=True) # upload to AWG
        print(f'Completed {t+1} iterations.')
    
    except KeyboardInterrupt:
        print(f'\nInterrupted at iteration {t}.')
        pass # allow the user to interrupt the simulation

    #print_final_energies(energies, og_betas, target_energy)
    e_history = np.array(e_history)
    if is_plotting:
        plot_hp_convergence(e_history, qubo_bits, og_betas, target_energy)
    # ask user whether to save the results
    if is_saving:
        is_save = input('Save results? (y/N): ')
        if is_save.lower() == 'y':
            save_results(model, e_history, bits_history, x_vector, og_betas, noise_std, asymmetric_J)

if __name__ == '__main__':
    model = load_hp_model_by_name('S6', latdim=(4,3))
    # solve_hp_isingmachine(model, num_iterations=100, num_ics=10, betas=(0.1, 0.01, 0.08), noise_std=0.09, asymmetric_J=True)
    solve_hp_isingmachine(
        model,
        num_iterations=15_000,
        num_ics=1000,
        betas=(0.001, 0.002, 0.003, 0.004),
        noise_std=0.185,
        asymmetric_J=True,
        is_plotting=True,
        is_saving=False,
        )
    
