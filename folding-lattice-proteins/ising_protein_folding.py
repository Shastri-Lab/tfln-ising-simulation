from hp_lattice import *
from math import ceil, floor
from dimod.utilities import qubo_to_ising
from dwave.samplers import SimulatedAnnealingSampler
from dwave_to_isingmachine import (
    get_idx_vector,
    flatten_spin_matrix,
    vector_to_spin_matrix,
    J_dict_to_mat,
    h_dict_to_mat,
    save_qubo_model_to_ising_mat,
)

ROOT2 = np.sqrt(2)
def sigma(x):
    return np.tanh(ROOT2*x) # TODO: do a cos^2 instead of this... this is just more idealized so easier to start with

def setup_hp_isingmachine(seq, latdim=None, savemat=False):
    print(f"Sequence {seq['name']}: {seq['sequence']}")
    print(f"Target Minimum Energy: {seq['min_energy']}")
    print("Generating QUBO model...")
    sequence = seq['sequence']
    seq_len = len(sequence)
    if latdim is None:
        lattice_size_x = 10
        lattice_size_y = 10
    else:
        lattice_size_x, lattice_size_y = latdim
    target_energy = seq['min_energy']
    model = Lattice_HP_QUBO(
        dim = [lattice_size_y, lattice_size_x],
        sequence=sequence,
        Lambda=(2.1, 2.4, 3.0),
    )
    Q = model.interaction_matrix()
    print("Converting QUBO to Ising model...")
    h_dict, J_dict, offset_ising = qubo_to_ising(Q)

    idx_vector = get_idx_vector(seq_len, lattice_size_y, lattice_size_x)
    h = h_dict_to_mat(h_dict, idx_vector)
    J = J_dict_to_mat(J_dict, idx_vector)
    
    if savemat:
        mat_filename = f'mat_files/{seq["name"]}_{lattice_size_y}x{lattice_size_x}.mat'
        print(f"Saving Ising model to MATLAB file {mat_filename}...", end="")
        save_qubo_model_to_ising_mat(model, mat_filename, target_energy=target_energy)
        print("done.")

    return h, J, model, target_energy

def solve_hp_isingmachine(h, J, model, target_energy):
    print("Solving using Ising Machine...", end="")
    
    num_spins = h.shape[0]
    x_vector = np.random.uniform(-1, 1, num_spins)

    alpha = 1.0
    beta = 0.4
    beta /= np.max(np.abs(J)) # normalize beta by the maximum coupling strength

    W = alpha * np.eye(num_spins) - beta * J
    b = beta * h

    x_history = []
    e_history = []

    num_iterations = 1000
    for t in range(num_iterations):        
        # every iteration
        spin_vector = np.sign(x_vector)
        qubo_bits = (spin_vector+1)/2
        
        # compute the energy of the current state
        energies = model.get_energies(qubo_bits)
        current_energy = sum(energies)

        # record the history
        x_history.append(x_vector)
        e_history.append(current_energy)
        
        # compute the next state of the system
        result = np.dot(W, sigma(x_vector))
        result += np.random.normal(0, 0.01, num_spins)
        result += b
        x_vector += result
    print("done. Plotting results...")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title('Ising Energy')
    ax[0].axhline(target_energy, color='r', linestyle='--')
    ax[0].plot(list(range(len(e_history))), e_history, color='b')
    ax[1].set_title('Final Lattice Configuration')
    model.show_lattice(qubo_bits, axes=ax[1])
    plt.show()

if __name__ == '__main__':
    import json
    with open('protein_sequences.json', 'r') as f:
        hp_sequences = json.load(f)

    # try S4 sequence first
    s4_seq = [seq for seq in hp_sequences if seq['name'] == 'S4'][0]
    params = setup_hp_isingmachine(s4_seq, latdim=(3,3), savemat=False)
    solve_hp_isingmachine(*params)
