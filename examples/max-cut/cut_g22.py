#%%
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

vis_graph = False

#%%
mat = scipy.io.loadmat('graphs/g22/G22.mat')
g22_sparse_csc = mat['Problem'][0,0][1]
g22 = g22_sparse_csc.toarray() # g22_sparse_csc is a `scipy.sparse._csc.csc_matrix` object, convert it to a numpy array


# %%

# visualize the graph
G22 = nx.from_numpy_array(g22)
if vis_graph:
    nx.draw(G22, with_labels=False, node_color='red', node_size=0.5, edge_color=(0.,0.,0.,0.1))
    plt.show()

# %%

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


# %%

best_g22_cut = 13_359

def cut_value(cut, J):
    # cut is the results of the spin bits history
    # it has shape (num_iterations, num_pars, num_ics, num_spins)
    cut_value = np.zeros((cut.shape[0], cut.shape[1], cut.shape[2])) # one cut value for each set of spins
    for it in range(cut.shape[0]):
        for par in range(cut.shape[1]):
            for ic in range(cut.shape[2]):
                value = 0
                for row, i in enumerate(cut[it, par, ic, :]):
                    for col, j in enumerate(cut[it, par, ic, :]):
                        if i != j: # only count edges between different partitions
                            value += J[row][col]
                cut_value[it, par, ic] = value
    
    return cut_value / 2

def cut_graph(
        graph,
        num_iterations=500,
        num_ics=100,
        alphas=None,
        betas=(0.1,),
        noise_std=0.1,
        simulated_annealing=False,
        early_break=True,
        sparse=False,
        ):

    adjacency_matrix = nx.adjacency_matrix(graph).toarray()

    min_energy = 4 * (0.25*np.sum(adjacency_matrix) - best_g22_cut)

    num_spins = adjacency_matrix.shape[0]
    J = adjacency_matrix
    h = np.zeros(num_spins)
    e_offset = 0

    problem = IsingProblem(J=J, h=h)
    config = SolverConfig(
        target_energy=min_energy,
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=alphas,
        betas=betas,
        noise_std=noise_std,
        simulated_annealing=simulated_annealing,
        early_break=early_break,
        sparse=sparse,
    )

    results = solve_isingmachine(problem, config)
    spin_vector = np.sign(results.final_vector)

    # find index of the minimum energy
    final_energies = results.energy_history[-1, :, :]
    min_energy_idx_flat = np.argmin(final_energies)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, final_energies.shape)
    min_energy = final_energies[min_energy_idx]
    min_spins = spin_vector[*min_energy_idx, :]


    fig, ax = plot_energy_convergence(results.energy_history, config, 'Max Cut Energy')
    
    cut = 0.25*np.sum(J) - 0.25*results.energy_history # energy is really 2E?
    # cut = cut_value(np.array(results.spin_bits_history), J)
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

    ax2.axhline(best_g22_cut, color='k', linestyle=(0, (1, 1)))

    print(f'Best cut value found is: {np.max(cut):.1f}')
    print(f'Reference cut value is: {best_g22_cut}')
    print(f'Got to {100*np.max(cut)/best_g22_cut:.2f}% of the reference cut value')

    plt.show()

if __name__ == '__main__':
    cut_graph(
        G22,
        num_ics=10,
        sparse=False, # sparse energy calculation needs optimization; it's about the same speed as dense
        early_break=True,
        betas=(0.08, 0.06, 0.04),
        noise_std=0.5,
        simulated_annealing=True,
        num_iterations=1600,
        )
