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

BEST_G22_CUT = 13_359

def cut_graph(
        graph,
        bit_precision,
        num_iterations=500,
        num_ics=100,
        alpha_search=(0,1),
        beta_search=(0,1),
        noise_std=0.5,
        early_break=True,
        sparse=False,
        ):

    adjacency_matrix = nx.adjacency_matrix(graph).toarray()

    min_energy = 4 * (0.25*np.sum(adjacency_matrix) - BEST_G22_CUT)

    num_spins = adjacency_matrix.shape[0]
    J = adjacency_matrix
    h = None # np.zeros(num_spins)
    e_offset = 0

    def annealing_schedule(t):
        return noise_std * 0.95**(t // 10)
    
    # find the possible alphas and betas to search over based on resolution
    
    levels = 2**bit_precision
    possible_values = np.linspace(-1, 1, levels, endpoint=True)
    alphas = possible_values[np.logical_and(possible_values >= alpha_search[0], possible_values <= alpha_search[1])]
    betas = possible_values[np.logical_and(possible_values >= beta_search[0], possible_values <= beta_search[1])]

    print(f'Searching over {len(alphas)} alphas and {len(betas)} betas')

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
        early_break=early_break,
        sparse=sparse,
        bit_precision=bit_precision,
    )

    results = solve_isingmachine(problem, config)
    spin_vector = np.sign(results.final_vector)

    filename = f'{os.path.dirname(__file__)}/results/g22_cut_sweep_{bit_precision}bit_3.npy'
    np.save(filename, results)
    print(f'Saved results to {filename}')

    # find index of the minimum energy
    final_energies = results.energy_history[-1, :, :]
    # min_energy_idx_flat = np.argmin(final_energies)
    # min_energy_idx = np.unravel_index(min_energy_idx_flat, final_energies.shape)
    # min_energy = final_energies[min_energy_idx]
    # min_spins = spin_vector[*min_energy_idx, :]

    average_final_energies = np.mean(final_energies, axis=-1) # average final energy for each parameter
    average_lowest_par_idx = np.argmin(average_final_energies)
    best_alpha, best_beta = config.alpha_beta[average_lowest_par_idx, :]
    final_cut_vals = 0.25*np.sum(J) - 0.25*final_energies[average_lowest_par_idx, :].squeeze() # TODO: energy is really 2E?
    avg_cuts = np.mean(final_cut_vals)

    print(f"For {bit_precision}-bit resoltuion, found {best_alpha=} and {best_beta=}, which have average cut {avg_cuts}")

    fig, ax = plot_energy_convergence(results.energy_history, config, 'Max Cut Energy')
    
    cut = 0.25*np.sum(J) - 0.25*results.energy_history # energy is really 2E?
    ax2 = ax.twinx()
    ax2.set_ylabel('Cut Value')

    num_ics = cut.shape[2]
    iters = list(range(cut.shape[0]))

    for i, (alpha, beta) in enumerate(config.alpha_beta):
        if num_ics > 1:
            ax2.fill_between(iters, cut[:, i, :].min(axis=-1), cut[:, i, :].max(axis=-1), alpha=0.1, color=f'C{i}')
        # figure out which one has max value at the last iteration
        max_cut_beta_idx = np.argmax(cut[-1, i, :])
        ax2.plot(iters, cut[:, i, max_cut_beta_idx], '--', lw=0.75, ms=2.0, label=f'α = {alpha:.1e}, β = {beta:.1e}', color=f'C{i}')

    ax2.axhline(BEST_G22_CUT, color='k', linestyle=(0, (1, 1)))

    print(f'Best cut value found is: {np.max(cut):.1f}')
    print(f'Reference cut value is: {BEST_G22_CUT}')
    print(f'Got to {100*np.max(cut)/BEST_G22_CUT:.2f}% of the reference cut value')

    plt.show()

if __name__ == '__main__':
    vis_graph = False

    mat = scipy.io.loadmat('graphs/g22/G22.mat')
    g22_sparse_csc = mat['Problem'][0,0][1]
    g22 = g22_sparse_csc.toarray() # g22_sparse_csc is a `scipy.sparse._csc.csc_matrix` object, convert it to a numpy array

    # visualize the graph
    G22 = nx.from_numpy_array(g22)
    if vis_graph:
        nx.draw(G22, with_labels=False, node_color='red', node_size=0.5, edge_color=(0.,0.,0.,0.1))
        plt.show()

    # print the list of distinct edge weights
    edge_weights = np.unique(g22)
    print(f'distinct edge weights: {edge_weights}')

    # check if it's a symmetric matrix
    print(f'is symmetric: {np.all(g22 == g22.T)}')

    # calculate the average degree
    degree = np.sum(g22, axis=0)
    avg_degree = np.mean(degree)
    print(f'average connections per node: {avg_degree}')

    # calulate the number of edges
    num_edges = np.sum(degree) / 2
    print(f'total num edges: {num_edges}')

    cut_graph(
        G22,
        bit_precision=1,
        num_ics=100,
        num_iterations=1000,
        sparse=False, # sparse energy calculation needs optimization; it's about the same speed as dense
        early_break=False,
        alpha_search=(0.999999, 1),   # 0.32
        beta_search=(0.5, 1),    # 3.2e-5 – 5.6e-6
        noise_std=1.0,
        )
    
    # cut_graph(
    #     G22,
    #     num_ics=10,
    #     num_iterations=500,
    #     sparse=False, # sparse energy calculation needs optimization; it's about the same speed as dense
    #     early_break=False,
    #     alphas=np.linspace(0.005, 0.205, 5, endpoint=True), # 0.32
    #     betas=np.logspace(-4, -1, 20, endpoint=True),    # 3.2e-5 – 5.6e-6
    #     noise_std=1.0,
    #     bit_precision=2,
    #     )



"""
recap of results filename suffixes:
1. no suffix: 10 trials, 500 iterations, 2^B+1 levels in W
2. _1: 100 trials, 1000 iterations, 2^B+1 levels in W
3. _2: 100 trials, 1000 iterations, 2^B levels in W, quantizing after nonlinearity
"""

