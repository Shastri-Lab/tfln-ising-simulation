import os
import json
import os.path as path
from hp_lattice import *
from math import ceil, floor
from dataclasses import dataclass
from dwave.samplers import SimulatedAnnealingSampler
from dwave_to_isingmachine import (
    flatten_spin_matrix,
    vector_to_spin_matrix,
    J_dict_to_mat,
    h_dict_to_mat,
    save_model_for_matlab,
)

ROOT2 = np.sqrt(2)
def sigma(x):
    return np.tanh(ROOT2*x) # TODO: do a cos^2 instead of this... this is just more idealized so easier to start with

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
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title('Ising Energy')
    if target_energy is not None:
        ax[0].axhline(target_energy, color='r', linestyle='--')
    
    iters = list(range(e_history.shape[0]))
    for i, beta in enumerate(betas):
        ax[0].plot(iters, e_history[:, i], label=f'β = {beta:.1e}')
    
    ax[0].legend()
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Energy')
    
    final_energy = e_history[-1, :]
    # find index of the minimum energy
    min_energy_idx = np.argmin(final_energy)
    min_energy = final_energy[min_energy_idx]
    min_qubo_bits = qubo_bits[min_energy_idx, :]
    min_energy_beta = betas[min_energy_idx]
    
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

def solve_hp_isingmachine(model, num_iterations=250_000, betas=0.005, noise_std=0.125):
    target_energy = model.target_energy
    h_dict, J_dict, ising_e_offset = model.to_ising()
    h = h_dict_to_mat(h_dict, model.keys)
    J = J_dict_to_mat(J_dict, model.keys)
    save_mat_file(model)

    betas = np.atleast_1d(betas)
    betas /= np.max(np.abs(J)) # normalize beta by the maximum coupling strength
    alphas = 1 - betas
    
    num_spins = h.shape[0]
    num_betas = betas.shape[0]

    W = np.zeros((num_betas, num_spins, num_spins))
    b = np.zeros((num_betas, num_spins))
    for i, (alpha, beta) in enumerate(zip(alphas, betas)):
        W[i, :, :] = alpha * np.eye(num_spins) - beta * J
        b[i, :] = -beta * h * 0.7

    x_init = np.random.uniform(-1, 1, num_spins)
    x_vector = np.vstack([x_init for _ in range(num_betas)])

    x_history = []
    e_history = []

    for t in range(num_iterations):
        spin_vector = np.sign(x_vector)     # σ ∈ {-1, 1}
        qubo_bits = (spin_vector+1)/2       # q ∈ {0, 1}
        
        # compute the energy of the current state
        energies = np.array([model.get_energies(qubo_bits[i, :]) for i in range(num_betas)])
        current_energy = np.sum(energies, axis=1)

        # record the history
        x_history.append(x_vector.copy())
        e_history.append(current_energy)
        
        # break if we are close enough to the target energy
        # if np.abs(current_energy - target_energy) < 1e-3:
        #     break
        # if np.any(np.abs(current_energy - target_energy) < 1e-3):
        #     break

        # compute the next state of the system
        output = np.einsum('ijk,ik->ij', W, sigma(x_vector))
        noise = np.random.normal(0, noise_std, num_spins)
        output += np.vstack([noise for _ in range(num_betas)])
        output += b
        x_vector = output

    print(f"done. final energy = {energies}")
    plot_hp_convergence(np.array(e_history), qubo_bits, betas, target_energy)

if __name__ == '__main__':
    model = load_hp_model_by_name('S4', latdim=(3,3))
    solve_hp_isingmachine(model, num_iterations=100_000, betas=(0.001, 0.005, 0.01, 0.05))
