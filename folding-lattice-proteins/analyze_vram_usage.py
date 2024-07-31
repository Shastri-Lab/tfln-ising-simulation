import torch
import matplotlib.pyplot as plt
import numpy as np
from ising_protein_folding import load_hp_model_by_name, solve_hp_problem_gpu
import itertools

def measure_vram_usage(num_ics, num_iterations, num_betas, lattice_size):
    """
    Measure VRAM usage for a given problem size.
    """    
    betas = np.logspace(-4, 0, num_betas)

    torch.cuda.reset_max_memory_allocated()
    
    # run the solver
    solve_hp_problem_gpu(
        load_hp_model_by_name(
            'S10',
            latdim=(lattice_size, lattice_size),
            lambdas=(2.1, 2.4, 3.0),
            ),
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=None, 
        betas=betas,
        noise_std=0.3,
        is_plotting=False,
        is_saving=False,
        )    
    
    vram_usage = torch.cuda.max_memory_allocated()
    
    return vram_usage


def vram_study():
    """
    Study VRAM usage as a function of problem size.
    """

    num_ics = np.logspace(1, 4, 4).astype(int)
    num_iterations = np.logspace(1, 3, 3).astype(int)
    num_betas = np.logspace(0, 3, 4).astype(int)
    lattice_sizes = [4, 5, 6]

    # iterate over all combinations of parameters
    vram_usages = {}
    for num_ic, num_iteration, num_beta, lattice_size in itertools.product(num_ics, num_iterations, num_betas, lattice_sizes):
        vram_usage = measure_vram_usage(num_ic, num_iteration, num_beta, lattice_size)
        vram_usages[(num_ic, num_iteration, num_beta, lattice_size)] = vram_usage

    return vram_usages, (num_ics, num_iterations, num_betas, lattice_sizes)

if __name__ == '__main__':
    vram_usages, (num_ics, num_iterations, num_betas, lattice_sizes) = vram_study()

    # make a plot with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # on each subplot, plot the VRAM usage as a function of a given parameter

    # plot VRAM usage as a function of number of initial conditions
    axs[0, 0].plot(num_ics, [vram_usages[(num_ic, num_iterations[0], num_betas[0], lattice_sizes[0])] for num_ic in num_ics])
    axs[0, 0].set_title('VRAM usage vs. Number of Initial Conditions')
    axs[0, 0].set_xlabel('Number of Initial Conditions')
    axs[0, 0].set_ylabel('VRAM Usage (bytes)')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')

    # plot VRAM usage as a function of number of iterations
    axs[0, 1].plot(num_iterations, [vram_usages[(num_ics[0], num_iteration, num_betas[0], lattice_sizes[0])] for num_iteration in num_iterations])
    axs[0, 1].set_title('VRAM usage vs. Number of Iterations')
    axs[0, 1].set_xlabel('Number of Iterations')
    axs[0, 1].set_ylabel('VRAM Usage (bytes)')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')

    # plot VRAM usage as a function of number of betas
    axs[1, 0].plot(num_betas, [vram_usages[(num_ics[0], num_iterations[0], num_beta, lattice_sizes[0])] for num_beta in num_betas])
    axs[1, 0].set_title('VRAM usage vs. Number of Betas')
    axs[1, 0].set_xlabel('Number of Betas')
    axs[1, 0].set_ylabel('VRAM Usage (bytes)')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')

    # plot VRAM usage as a function of lattice size
    axs[1, 1].plot(lattice_sizes, [vram_usages[(num_ics[0], num_iterations[0], num_betas[0], lattice_size)] for lattice_size in lattice_sizes])
    axs[1, 1].set_title('VRAM usage vs. Lattice Size')
    axs[1, 1].set_xlabel('Lattice Size')
    axs[1, 1].set_ylabel('VRAM Usage (bytes)')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()
    
