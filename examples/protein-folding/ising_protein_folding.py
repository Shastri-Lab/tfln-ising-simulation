import os
import json
import numpy as np
from tqdm import tqdm
import os.path as path
import matplotlib.pyplot as plt
from hp_lattice import Lattice_HP_QUBO
from dimod.utilities import qubo_to_ising

from pysing_machine.core.ising_solver import (
    IsingProblem,
    SolverConfig,
    solve_isingmachine,
    solve_isingmachine_gpu,
)
from pysing_machine.core.utils import (
    J_dict_to_mat,
    h_dict_to_mat,
    save_model_for_matlab,
    save_results,
)
from pysing_machine.core.plotting import plot_energy_convergence

    
def load_hp_model_by_name(name, latdim=(10,10), lambdas=(2.1, 2.4, 3.0)):
    """
    Load a protein folding QUBO model by name.
    """
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

# def plot_hp_convergence(model, e_history, qubo_bits, alpha_beta, noise_std, target_energy=None):
def plot_hp_convergence(model, e_history, config, final_state):
    title = f'{model.name} ({model.seq_to_str()})\n{model.dim[1]}x{model.dim[0]} lattice\nNoise std={config.noise_std}'
    fig, ax = plot_energy_convergence(e_history, config, title)
    
    # find index of the minimum energy
    total_energy_hist = e_history.sum(axis=1) if e_history.ndim == 4 else e_history
    final_energies = total_energy_hist[-1, :, :]
    min_energy_idx_flat = np.argmin(final_energies)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, final_energies.shape)
    min_energy = final_energies[min_energy_idx]
    min_qubo_bits = final_state[*min_energy_idx, :]
    min_energy_alpha, min_energy_beta = config.alpha_beta[min_energy_idx[0]]
    # print(f'Minimum bits has sum: {min_qubo_bits.sum()}')

    # create inset for lattice
    inset_ax = fig.add_axes([0.75, 0.625, 0.155, 0.3])
    inset_ax.set_title(f'E = {min_energy:.1f}\nα = {min_energy_alpha:.1e}, β = {min_energy_beta:.1e}', horizontalalignment='left', loc='left')
    model.show_lattice(min_qubo_bits, axes=inset_ax)
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

def solve_hp_problem(model, num_iterations=250_000, num_ics=2, alphas=None, betas=0.005, noise_std=0.125, early_break=True, is_plotting=True, is_saving=True, simulated_annealing=False, annealing_iters=1, annealing_fraction=1.0, make_symmetric=False, sparse=False):
    print(f'\nSetting up {model.name} simulation on {model.dim[1]}x{model.dim[0]} lattice...')
    print('Coverting QUBO to Ising...')
    save_mat_file(model)
    target_energy = model.target_energy
    h_dict, J_dict, ising_e_offset = model.to_ising()
    h = h_dict_to_mat(h_dict, model.keys)
    J = J_dict_to_mat(J_dict, model.keys)
    ising_e_offset += model.Lambda[0]*model.len_of_seq


    def modified_exp_schedule(t):
        return noise_std * annealing_fraction**(t//annealing_iters)

    problem = IsingProblem(
        J=J,
        h=h,
        e_offset=ising_e_offset,
    )
    config = SolverConfig(
        target_energy=target_energy,
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=alphas,
        betas=betas,
        start_temperature=noise_std,
        annealing_schedule="custom",
        custom_schedule=modified_exp_schedule,
        make_symmetric=make_symmetric,
        sparse=sparse,
        early_break=early_break,
    )

    results = solve_isingmachine(problem, config)
    x_vector, bits_history, e_history = results.final_vector, results.spin_bits_history, results.energy_history

    if is_plotting:
        plot_hp_convergence(model, e_history, config, bits_history[-1])
    if is_saving:
        is_save = input('Save results? (y/N): ')
        if is_save.lower() == 'y':
            save_results(model, e_history, bits_history, x_vector, config.alpha_beta, noise_std)

def solve_hp_problem_gpu(model, num_iterations=250_000, num_ics=2, alphas=None, betas=0.005, noise_std=0.125, is_plotting=False, is_saving=False, early_break=True, skip_energy=False):
    print(f'\nSetting up {model.name} simulation on {model.dim[1]}x{model.dim[0]} lattice...')
    print('Coverting QUBO to Ising...')
    save_mat_file(model)
    target_energy = model.target_energy
    h_dict, J_dict, ising_e_offset = model.to_ising()
    h = h_dict_to_mat(h_dict, model.keys)
    J = J_dict_to_mat(J_dict, model.keys)
    ising_e_offset += model.Lambda[0] * model.len_of_seq

    x_vector, bits_history, e_history, alpha_beta, qubo_bits = solve_isingmachine_gpu(
        J,
        h,
        e_offset=ising_e_offset,
        target_energy=target_energy,
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=alphas,
        betas=betas,
        noise_std=noise_std,
        early_break=early_break,
        save_iter_freq=5,
        skip_energy_calculation=skip_energy,
    )

    if is_plotting:
        plot_hp_convergence(model, e_history, qubo_bits, alpha_beta, noise_std, target_energy)
    if is_saving:
        is_save = input('Save results? (y/N): ')
        if is_save.lower() == 'y':
            save_results(model, e_history, bits_history, x_vector, alpha_beta, noise_std)


