import numpy as np

def energy(spins, adjacency_matrix):
    return spins @ adjacency_matrix @ spins

def get_min_energy(adjacency_matrix):
    min_energy_spin_vector = np.ones(adjacency_matrix.shape[0])
    min_energy_spin_vector[1::2] = -1
    return energy(min_energy_spin_vector, adjacency_matrix)