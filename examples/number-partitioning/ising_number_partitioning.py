import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from matrix_generation import generate_even_sum_set, ising_number_partitioning_matrix
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



def plot_set_sums(nset, bits_history, config, title=None):
    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.subplots_adjust(left=0.05, right=0.825)  # Adjust right to make space for the legend
    if title:
        fig.suptitle(title, horizontalalignment='left', x=0.1)

    ax.set_title('Subset Difference')
    
    # e_history has shape (T, B, I) or (T, 4, B, I) if energies are separated
    other_energies = []

    num_ics = config.num_ics
    iters = list(range(config.num_iterations))
    spins_history = np.array(bits_history[:-1]).astype(np.float32)*2-1


    for i, (alpha, beta) in enumerate(config.alpha_beta):
        # figure out which one has minimum energy at the last iteration
        for j in range(num_ics):
            spins = spins_history[:, i, j, :]
            difference = np.abs((nset * spins).sum(axis=1))
            num_zeros = difference.shape[0] - np.count_nonzero(difference)
            print(f'α = {alpha:.1e}, β = {beta:.1e}, IC {j}: {num_zeros} zeros')
            ax.plot(iters, difference, '.-', lw=0.75, ms=2.0, label=f'α = {alpha:.1e}, β = {beta:.1e}', color=f'C{i}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy')
    
    # move legend outside of the plot
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    
    return fig, ax

def solve_number_partitioning_mat(mat_file):
    import scipy.io
    mat = scipy.io.loadmat(mat_file)
    J = mat['J']
    nset = mat['nset'][0]
    h = np.zeros(J.shape[0])
    
    problem = IsingProblem(J=J, h=h)
    config = SolverConfig(
        target_energy=None, # -(nset**2).sum(),
        num_iterations=100,
        num_ics=20,
        alphas=None,
        betas=(0.005, 0.01, 0.02),
        noise_std=0.15,
        simulated_annealing=False,
    )
    results = solve_isingmachine(problem, config)
    x_vector, bits_history, e_history = results.final_vector, results.spin_bits_history, results.energy_history
    
    f1, a1 = plot_energy_convergence(e_history, config)
    f2, a2 = plot_set_sums(nset, bits_history, config, title='Number Partitioning Sets')
    f1_manager = plt.figure(f1.number).canvas.manager
    f1_manager.window.wm_geometry("+0+0")
    f2_manager = plt.figure(f2.number).canvas.manager
    f2_manager.window.wm_geometry("+1000+600")
    plt.show()


def solve_ising_number_partitioning(size, max_value=16, num_iterations=1000, num_ics=1, alphas=None, betas=0.005, noise_std=0.125, simulated_annealing=False, is_plotting=False):
    nset = generate_even_sum_set(size, max_value)
    J = ising_number_partitioning_matrix(nset)
    h = np.zeros(size)
    
    x_vector, bits_history, e_history, alpha_beta, qubo_bits = solve_isingmachine(
        J,
        h,
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=alphas,
        betas=betas,
        noise_std=noise_std,
        simulated_annealing=simulated_annealing,
    )
    qubo_bits = qubo_bits[-1, -1, :] # TODO: this is only one value. need to find min energy solution
    subset1 = nset[qubo_bits == 0]
    subset2 = nset[qubo_bits == 1]
    
    print(f'The set is: {list(nset)}')
    print(f'Subset 1: {list(subset1)}')
    print(f'Subset 2: {list(subset2)}')
    print(f'Sum of Subset 1: {np.sum(subset1)}')
    print(f'Sum of Subset 2: {np.sum(subset2)}')
    print(f'Difference between sums: {abs(np.sum(subset1) - np.sum(subset2))}')

    if is_plotting:
        plot_energy_convergence(e_history,alpha_beta, noise_std)
    # if is_saving:
    #     is_save = input('Save results? (y/N): ')
    #     if is_save.lower() == 'y':
    #         save_results(model, e_history, bits_history, x_vector, alpha_beta, noise_std)


if __name__ == '__main__':
    solve_number_partitioning_mat('mat_files/numberPartitioning_256N.mat')
    # solve_ising_number_partitioning(16, 16, is_plotting=True)
    # plt.show()
