#%%
import os
import scipy.io
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pysing_machine.core.ising_solver import (
    IsingProblem,
    SolverConfig,
    solve_isingmachine,
)
from pysing_machine.core.plotting import plot_energy_convergence

#%%
mat = scipy.io.loadmat('graphs/g22/G22.mat')
g22_sparse_csc = mat['Problem'][0,0][1]
g22 = g22_sparse_csc.toarray() # g22_sparse_csc is a `scipy.sparse._csc.csc_matrix` object, convert it to a numpy array
G22 = nx.from_numpy_array(g22)
best_g22_cut = 13_359 # this is the best known cut value for this graph

# %%

adjacency_matrix = nx.adjacency_matrix(G22).toarray()
min_energy = 4 * (0.25*np.sum(adjacency_matrix) - best_g22_cut)
num_spins = adjacency_matrix.shape[0]
J = adjacency_matrix
h = None # np.zeros(num_spins)
e_offset = 0

def cut_graph_quantized(
        bit_resolution,
        num_iterations=1000,
        num_ics=100,
        alphas=None,
        betas=(0.05,),
        noise_std=1.0,
        sparse=False,
        ):


    def annealing_schedule(t):
        return noise_std * 0.95**(t//10)

    problem = IsingProblem(J=J, h=h)
    config = SolverConfig(
        target_energy=min_energy,
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=alphas,
        betas=betas,
        start_temperature=noise_std,
        annealing_schedule="custom",
        custom_schedule=annealing_schedule, 
        early_break=False,
        sparse=sparse,
        bit_precision=bit_resolution,
    )

    results = solve_isingmachine(problem, config)
    spin_vector = np.sign(results.final_vector)

    final_energies = results.energy_history[-1, :, :] # has shape (num_pars, num_ics)
    
    average_final_energies = np.mean(final_energies, axis=-1) # average final energy for each parameter
    average_lowest_par_idx = np.argmin(average_final_energies)
    final_energies = final_energies[average_lowest_par_idx, :].squeeze()

    final_cut_vals = 0.25*np.sum(J) - 0.25*final_energies # TODO: energy is really 2E?

    return final_cut_vals


if __name__ == '__main__':
    pwd = os.path.dirname(__file__)
    filename = f'{pwd}/results/g22_cut_vs_bit_resolution_plus1.npy'
    
    best_alphas = {
        1: 1.0,
        2: 1.0,                                                                          # 0.205 * np.linspace(0.9, 1.1, 10),
        3: 1.0,                                                                          # 0.21 * np.linspace(0.9, 1.1, 10),
        4: 1.0,                                                                         # 0.15000000000000002 * np.linspace(0.9, 1.1, 10),
        5: 1.0,                                                                         # 0.15000000000000002 * np.linspace(0.9, 1.1, 10),
        6: 1.0,                                                                          # 0.2 * np.linspace(0.9, 1.1, 10),
        7: 1.0,                                                                         # 0.2 * np.linspace(0.9, 1.1, 10),
        8: 1.0,                                                                         # 0.2 * np.linspace(0.9, 1.1, 10), 
    }
    best_betas = {
        1: 1.0,
        2: 1/3,                                                                          # 0.0001 * np.linspace(0.9, 1.1, 10),                  # 82.09 %
        3: 0.1428571428571428,                                                     # 0.029763514416313176 * np.linspace(0.9, 1.1, 10),    # 97.33 %
        4: 2/30,                                                                          # 0.01 * np.linspace(0.9, 1.1, 10),                    # 98.17 %
        5: 0.032258064516129004,                                                                          # 0.01 * np.linspace(0.9, 1.1, 10),                    # 98.56 %
        6: 0.04761904761904745,                                                                          # 0.01 * np.linspace(0.9, 1.1, 10),                    # 98.87 %
        7: 0.055118110236220375,                                                                         # 0.01 * np.linspace(0.9, 1.1, 10),                    # 98.68 %
        8: 0.050980392156862786,                                                                         # 0.01 * np.linspace(0.9, 1.1, 10),                    # 98.65 %
    }
    bit_resolutions = list(best_alphas.keys())

    final_cuts = [cut_graph_quantized(res,
                                      alphas=(best_alphas[res],),
                                      betas=(best_betas[res],)
                                      ) for res in bit_resolutions]
    np.save(filename, {res: cuts for res, cuts in zip(bit_resolutions, final_cuts)})
    print(f'Saved results to {filename}')

