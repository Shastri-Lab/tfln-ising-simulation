import os
import json
import cProfile
import pstats
import numpy as np
from tqdm import tqdm
import os.path as path
from math import ceil, floor
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from dataclasses import dataclass
from hp_lattice import Lattice_HP_QUBO
from dimod.utilities import qubo_to_ising
from ising_machine import solve_isingmachine
from dwave.samplers import SimulatedAnnealingSampler
from dwave_to_isingmachine import (
    flatten_spin_matrix,
    vector_to_spin_matrix,
    J_dict_to_mat,
    h_dict_to_mat,
    save_model_for_matlab,
)

    
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

def plot_hp_convergence(model, e_history, qubo_bits, alpha_beta, noise_std, target_energy=None):
    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.subplots_adjust(left=0.05, right=0.825)  # Adjust right to make space for the legend
    fig.suptitle(f'{model.name} ({model.seq_to_str()})\n{model.dim[1]}x{model.dim[0]} lattice\nNoise std={noise_std}', horizontalalignment='left', x=0.1)

    ax.set_title('Ising Energy')
    if target_energy is not None:
        ax.axhline(target_energy, color='k', linestyle=(0, (1, 1)))
    
    # e_history has shape (T, B, I) or (T, 4, B, I) if energies are separated
    other_energies = []
    if e_history.ndim == 4:
        # separate into HP, C1, C2, C3
        e_hp = e_history[:, 0, :, :]
        e_c1 = e_history[:, 1, :, :]
        e_c2 = e_history[:, 2, :, :]
        e_c3 = e_history[:, 3, :, :]
        other_energies = [e_hp, e_c1, e_c2, e_c3]
        total_energy_hist = e_hp + e_c1 + e_c2 + e_c3
    else:
        total_energy_hist = e_history

    num_ics = total_energy_hist.shape[2]
    iters = list(range(total_energy_hist.shape[0]))
    for i, (alpha, beta) in enumerate(alpha_beta):
        if num_ics > 1: 
            ax.fill_between(iters, total_energy_hist[:, i, :].min(axis=-1), total_energy_hist[:, i, :].max(axis=-1), alpha=0.1, color=f'C{i}')
        # figure out which one has minimum energy at the last iteration
        min_energy_beta_idx = np.argmin(total_energy_hist[-1, i, :])
        ax.plot(iters, total_energy_hist[:, i, min_energy_beta_idx], '.-', lw=0.75, ms=2.0, label=f'α = {alpha:.1e}, β = {beta:.1e}', color=f'C{i}')
        for energy_component, ls in zip(other_energies, (':', '--', '-.', (0, (3, 5, 1, 5, 1, 5)))):
            ax.plot(iters, energy_component[:, i, min_energy_beta_idx], ls=ls, lw=1.0, color=f'C{i}')

    # make custom legend for the energy components
    ax.plot([], [], ls=':', color='k', label='$E_{HP}$')
    ax.plot([], [], ls='--', color='k', label='$E_{1}$')
    ax.plot([], [], ls='-.', color='k', label='$E_{2}$')
    ax.plot([], [], ls=(0, (3, 5, 1, 5, 1, 5)), color='k', label='$E_{3}$')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy')
    
    # move legend outside of the plot
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    
    # find index of the minimum energy
    final_energies = total_energy_hist[-1, :, :]
    min_energy_idx_flat = np.argmin(final_energies)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, final_energies.shape)
    min_energy = final_energies[min_energy_idx]
    min_qubo_bits = qubo_bits[*min_energy_idx, :]
    min_energy_alpha, min_energy_beta = alpha_beta[min_energy_idx[0]]
    print(f'Minimum bits has sum: {min_qubo_bits.sum()}')

    # create inset for lattice
    inset_ax = fig.add_axes([0.75, 0.625, 0.155, 0.3])
    inset_ax.set_title(f'E = {min_energy:.1f}\nα = {min_energy_alpha:.1e}, β = {min_energy_beta:.1e}', horizontalalignment='left', loc='left')
    model.show_lattice(min_qubo_bits, axes=inset_ax)
    plt.show()

def plot_hp_convergence_old(model, e_history, qubo_bits, alpha_beta, target_energy=None):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8.8))
    fig.suptitle(f'{model.name}')

    ax[0].set_title('Ising Energy')
    if target_energy is not None:
        ax[0].axhline(target_energy, color='k', linestyle='--')
    
    # e_history has shape (T, B, I)
    num_ics = e_history.shape[2]
    iters = list(range(e_history.shape[0]))
    for i, (alpha, beta) in enumerate(alpha_beta):
        if num_ics > 1: 
            ax[0].fill_between(iters, e_history[:, i, :].min(axis=-1), e_history[:, i, :].max(axis=-1), alpha=0.1, color=f'C{i}')
        # figure out which one has minimum energy at the last iteration
        min_energy_beta_idx = np.argmin(e_history[-1, i, :])
        ax[0].plot(iters, e_history[:, i, min_energy_beta_idx], '.-', lw=0.75, ms=2.0, label=f'α = {alpha:.1e}, β = {beta:.1e}', color=f'C{i}')
    
    ax[0].legend()
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Energy')
    
    final_energies = e_history[-1, :, :]
    # find index of the minimum energy
    min_energy_idx_flat = np.argmin(final_energies)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, final_energies.shape)
    min_energy = final_energies[min_energy_idx]
    min_qubo_bits = qubo_bits[*min_energy_idx, :]
    min_energy_alpha, min_energy_beta = alpha_beta[min_energy_idx[0]]
    
    ax[1].set_title(f'Final Lattice Configuration (α = {min_energy_alpha:.1e} β = {min_energy_beta:.1e}, E = {min_energy:.1f})')
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
        print(f'Saving matrices to {mat_filename}...', end=' ')
        save_model_for_matlab(model, mat_filename)
        print('Done.')

def print_final_energies(final_energies, betas, target_energy):
    print(f'Target Energy = {target_energy:.1f}\nFinal Energies:')
    for i, beta in enumerate(betas):
        energy = final_energies[i]
        energy = [round(e, 1) for e in energy]
        print(f'β = {beta:.1e}: E = {sum(energy):.1f} {energy}')

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
        sequence=model.sequence,
        target_energy=model.target_energy,
        lambdas=model.Lambda,
        noise_std=noise_std,
        )
    print('Done.')

def solve_hp_problem(model, num_iterations=250_000, num_ics=2, alphas=None, betas=0.005, noise_std=0.125, is_plotting=True, is_saving=True, simulated_annealing=False, separate_energies=False):
    print(f'\nSetting up {model.name} simulation on {model.dim[1]}x{model.dim[0]} lattice...')
    print('Coverting QUBO to Ising...')
    save_mat_file(model)
    target_energy = model.target_energy
    h_dict, J_dict, ising_e_offset = model.to_ising()
    h = h_dict_to_mat(h_dict, model.keys)
    J = J_dict_to_mat(J_dict, model.keys)

    if separate_energies:
        QHP = model.optimization_matrix()
        h_dict_HP, J_dict_HP, e_HP = qubo_to_ising(QHP)
        h_HP = h_dict_to_mat(h_dict_HP, model.keys)
        J_HP = J_dict_to_mat(J_dict_HP, model.keys)

        Q1 = model.constraint_matrix_1()
        h_dict_1, J_dict_1, e_1 = qubo_to_ising(Q1)
        h_1 = h_dict_to_mat(h_dict_1, model.keys)
        J_1 = J_dict_to_mat(J_dict_1, model.keys)

        Q2 = model.constraint_matrix_2()
        h_dict_2, J_dict_2, e_2 = qubo_to_ising(Q2)
        h_2 = h_dict_to_mat(h_dict_2, model.keys)
        J_2 = J_dict_to_mat(J_dict_2, model.keys)

        Q3 = model.constraint_matrix_3()
        h_dict_3, J_dict_3, e_3 = qubo_to_ising(Q3)
        h_3 = h_dict_to_mat(h_dict_3, model.keys)
        J_3 = J_dict_to_mat(J_dict_3, model.keys)

        J = [J_HP, J_1, J_2, J_3]
        h = [h_HP, h_1, h_2, h_3]
        ising_e_offset = [e_HP, e_1+model.Lambda[0]*model.len_of_seq, e_2, e_3]
    else:
        ising_e_offset += model.Lambda[0]*model.len_of_seq

    x_vector, bits_history, e_history, alpha_beta, qubo_bits = solve_isingmachine(
        J,
        h,
        e_offset=ising_e_offset,
        target_energy=target_energy,
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=alphas,
        betas=betas,
        noise_std=noise_std,
        simulated_annealing=simulated_annealing,
    )

    if is_plotting:
        plot_hp_convergence(model, e_history, qubo_bits, alpha_beta, noise_std, target_energy)
    if is_saving:
        is_save = input('Save results? (y/N): ')
        if is_save.lower() == 'y':
            save_results(model, e_history, bits_history, x_vector, alpha_beta, noise_std)

if __name__ == '__main__':

    is_profiling = False

    if is_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    hp_model = load_hp_model_by_name('S10', latdim=(4,3), lambdas=(10, 11, 12))
    # solve_hp_isingmachine(model, num_iterations=100, num_ics=10, betas=(0.1, 0.01, 0.08), noise_std=0.09, asymmetric_J=True)
    solve_hp_problem(
        hp_model,
        num_iterations=250,
        num_ics=50000,
        alphas=(0.1, ), # 0.85, 0.9, 0.999), # np.logspace(0, -0.25, 5),
        betas=(0.05, 0.005), # 0.0025, 0.05, 0.1), # np.logspace(-4, -0.25, 5),
        noise_std=0.04,
        is_plotting=True,
        is_saving=False,
        simulated_annealing=True,
        separate_energies=True,
        )
    
    if is_profiling:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)
