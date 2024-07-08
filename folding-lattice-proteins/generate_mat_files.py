from hp_lattice import *
from math import ceil, floor
from dimod.utilities import qubo_to_ising
from dwave.samplers import SimulatedAnnealingSampler
from dwave_to_isingmachine import (
    flatten_spin_matrix,
    vector_to_spin_matrix,
    J_dict_to_mat,
    h_dict_to_mat,
    save_qubo_model_to_ising_mat,
)

def setup_hp_isingmachine(seq, overwrite=False):
    sequence = seq['sequence']
    seq_len = len(sequence)
    lattice_size_x, lattice_size_y = seq.get('latdim', (10,10))
    lambdas = seq.get('lambdas', (2.1, 2.4, 3.0))
    target_energy = seq['min_energy']
    mat_filename = f'mat_files/{seq["name"]}_{lattice_size_y}x{lattice_size_x}.mat'
    
    try:
        with open(mat_filename, 'r') as f:
            pass
        if not overwrite:
            print(f"MATLAB file {mat_filename} already exists. Skipping...")
            return
    except FileNotFoundError:
        pass
    
    model = Lattice_HP_QUBO(
        dim = [lattice_size_y, lattice_size_x],
        sequence=sequence,
        Lambda=lambdas,
    )
    Q = model.interaction_matrix()
    h_dict, J_dict, offset_ising = qubo_to_ising(Q)

    idx_vector = model.keys
    h = h_dict_to_mat(h_dict, idx_vector)
    J = J_dict_to_mat(J_dict, idx_vector)
    
    print(f"Saving Ising model to MATLAB file {mat_filename}...", end="")
    save_qubo_model_to_ising_mat(model, mat_filename, target_energy=target_energy)
    print("done.")

    return h, J, model, target_energy

if __name__ == '__main__':
    import json
    with open('protein_sequences.json', 'r') as f:
        hp_sequences = json.load(f)

    for seq in hp_sequences:
        setup_hp_isingmachine(seq, overwrite=True)