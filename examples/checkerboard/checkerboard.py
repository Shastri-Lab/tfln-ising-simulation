import numpy as np
import matplotlib.pyplot as plt
from pysing_machine.core.ising_solver import (
    IsingProblem,
    SolverConfig,
    solve_isingmachine,
)
from pysing_machine.core.plotting import plot_energy_convergence

from plot_checkerboard import visualize_spins
from checkerboard_utils import get_min_energy
import networkx as nx

def plot_checkerboard_convergence(results, config, min_spins):
    lattice_size = int(np.sqrt(len(min_spins)))
    title = f'{lattice_size}x{lattice_size} checkerboard lattice\nNoise std={config.noise_std}'
    fig, ax = plot_energy_convergence(results.energy_history, config, title)

    inset_ax = fig.add_axes([0.75, 0.625, 0.155, 0.3])
    visualize_spins(min_spins, ax=inset_ax)
    plt.show()

def solve_checkerboard(
        checkerboard_size,
        num_iterations=500,
        num_ics=100,
        alphas=None,
        betas=(0.1,),
        noise_std=0.1,
        simulated_annealing=False,
        early_break=True,
        sparse=False,
        ):
    
    if checkerboard_size % 2 != 1:
        raise ValueError('Checkerboard size must be odd')

    num_spins = checkerboard_size * checkerboard_size

    graph = nx.grid_2d_graph(checkerboard_size, checkerboard_size)
    adjacency_matrix = nx.adjacency_matrix(graph).toarray()

    min_energy = get_min_energy(adjacency_matrix)

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

    plot_checkerboard_convergence(results, config, min_spins)


if __name__ == '__main__':
    solve_checkerboard(
        101,
        num_ics=1,
        sparse=True,
        early_break=True,
        betas=(0.5),
        noise_std=0.5,
        simulated_annealing=True,
        num_iterations=1_500,
        )