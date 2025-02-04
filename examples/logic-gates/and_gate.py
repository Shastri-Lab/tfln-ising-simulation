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
from pysing_machine.core.ising_solver import solve_isingmachine
from dwave.samplers import SimulatedAnnealingSampler
from pysing_machine.core.utils import (
    flatten_spin_matrix,
    vector_to_spin_matrix,
    J_dict_to_mat,
    h_dict_to_mat,
    save_model_for_matlab,
)


def plot_convergence(e_history, alpha_beta, target_energy=None):
    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.subplots_adjust(left=0.05, right=0.825)  # Adjust right to make space for the legend

    ax.set_title('Ising Energy')
    if target_energy is not None:
        ax.axhline(target_energy, color='k', linestyle='--')
    
    # e_history has shape (T, B, I)
    num_ics = e_history.shape[2]
    iters = list(range(e_history.shape[0]))
    for i, (alpha, beta) in enumerate(alpha_beta):
        if num_ics > 1: 
            ax.fill_between(iters, e_history[:, i, :].min(axis=-1), e_history[:, i, :].max(axis=-1), alpha=0.1, color=f'C{i}')
        # figure out which one has minimum energy at the last iteration
        min_energy_beta_idx = np.argmin(e_history[-1, i, :])
        ax.plot(iters, e_history[:, i, min_energy_beta_idx], '.-', lw=0.75, ms=2.0, label=f'α = {alpha:.1e}, β = {beta:.1e}', color=f'C{i}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy')
    
    # move legend outside of the plot
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    
    plt.show()


J = -np.array([
    [0, -1, 2,],
    [-1, 0, 2,],
    [2, 2, 0,],
])

h = -np.array([1, 1, -2])

desired_spins = np.ones(h.shape[0])
target_energy = desired_spins @ J @ desired_spins + desired_spins @ h

beta = 0.01
alpha = 1-beta
noise_std = 0.2

(x_vector,
 bits_history,
 e_history,
 alpha_beta,
 qubo_bits) = solve_isingmachine(
     J, h,
     e_offset=0.0,
     target_energy=None,
     num_iterations=1000,
     num_ics=100_000,
     alphas=alpha,
     betas=beta,
     noise_std=noise_std,
     early_break=False)


final_energies = e_history[-1, :, :]
min_energy_idx_flat = np.argmin(final_energies)
min_energy_idx = np.unravel_index(min_energy_idx_flat, final_energies.shape)
min_energy = final_energies[min_energy_idx]
min_qubo_bits = qubo_bits[*min_energy_idx, :]
# print(f'Minimum bits has sum: {min_qubo_bits.sum()}')
print(f'final: {min_qubo_bits}')

plot_convergence(e_history, alpha_beta, target_energy)