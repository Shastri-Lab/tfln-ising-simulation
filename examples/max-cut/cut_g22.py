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

def cut_graph(
        graph,
        num_iterations=500,
        num_ics=100,
        alphas=None,
        betas=(0.1,),
        noise_std=0.5,
        early_break=True,
        sparse=False,
        ):

    adjacency_matrix = nx.adjacency_matrix(graph).toarray()

    min_energy = 4 * (0.25*np.sum(adjacency_matrix) - best_g22_cut)

    num_spins = adjacency_matrix.shape[0]
    J = adjacency_matrix
    h = np.zeros(num_spins)
    e_offset = 0

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
        early_break=early_break,
        sparse=sparse,
    )

    results = solve_isingmachine(problem, config)
    spin_vector = np.sign(results.final_vector)

    # find index of the minimum energy
    final_energies = results.energy_history[-1, :, :]


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

    ax2.axhline(best_g22_cut, color='k', linestyle=(0, (1, 1)))

    print(f'Best cut value found is: {np.max(cut):.1f}')
    print(f'Reference cut value is: {best_g22_cut}')
    print(f'Got to {100*np.max(cut)/best_g22_cut:.2f}% of the reference cut value')

    plt.show()

if __name__ == '__main__':
    cut_graph(
        G22,
        num_ics=100,
        num_iterations=800,
        sparse=False,
        early_break=False,
        alphas=(1.0), #np.logspace(-0.1, 0, 3, endpoint=True),
        betas=(0.056), #np.logspace(-2, -0.5, 4, endpoint=False),
        noise_std=1.0,
        )
