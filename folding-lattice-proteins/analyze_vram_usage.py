#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
from ising_protein_folding import load_hp_model_by_name, solve_hp_problem_gpu
import matplotlib.cm as cm
import itertools
import datetime

def measure_vram_usage(num_ics, num_iterations, num_betas, lattice_size):
    """
    Measure VRAM usage for a given problem size.
    """    
    betas = np.logspace(-4, 0, num_betas)

    torch.cuda.reset_max_memory_allocated()
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
        early_break=False,
        )
    vram_usage = torch.cuda.max_memory_allocated()
    
    return vram_usage


def vram_study():
    """
    Study VRAM usage as a function of problem size.
    """

    num_ics = np.logspace(2, 4, 3).astype(int)
    num_iterations = [1, 2]
    num_betas = np.logspace(1, 3, 3).astype(int)
    lattice_sizes = [4, 5, 6]

    vram_usages = {}
    for num_ic, num_iteration, num_beta, lattice_size in itertools.product(num_ics, num_iterations, num_betas, lattice_sizes):
        try:
            vram_usage = measure_vram_usage(num_ic, num_iteration, num_beta, lattice_size)
        except torch.OutOfMemoryError:
            vram_usage = 12.0 * 2**30 # 12.0 GiB is capacity of 3060 GPU I'm using
        vram_usages[(num_ic, num_iteration, num_beta, lattice_size)] = vram_usage

    now = datetime.datetime.now()
    datetime_stamp = now.strftime("%Y%m%d%H%M")
    np.savez(f'results/vram_study_{datetime_stamp}.npz',
             vram_usages=vram_usages,
             num_ics=num_ics,
             num_iterations=num_iterations,
             num_betas=num_betas,
             lattice_sizes=lattice_sizes,
             )

    return vram_usages, (num_ics, num_iterations, num_betas, lattice_sizes)


def plot_study(vram_usages, num_ics, num_iterations, num_betas, lattice_sizes):
    # make a plot with 4 subplots
    fig, axs = plt.subplot_mosaic([['ics', 'iters'],
                                   ['pars', 'size']], figsize=(10, 10))

    # on each subplot, plot the VRAM usage as a function of a given parameter
    # plot VRAM usage as a function of number of initial conditions
    for i in range(len(num_iterations)):
        for j in range(len(num_betas)):
            for k in range(len(lattice_sizes)):
                axs['ics'].plot(num_ics, [vram_usages[(num_ic, num_iterations[i], num_betas[j], lattice_sizes[k])] for num_ic in num_ics], '.-')
    axs['ics'].set_title('VRAM usage vs. Number of Initial Conditions')
    axs['ics'].set_xlabel('Number of Initial Conditions')
    axs['ics'].set_ylabel('VRAM Usage (bytes)')
    axs['ics'].set_xscale('log')
    axs['ics'].set_yscale('log')

    # plot VRAM usage as a function of number of iterations
    for i in range(len(num_ics)):
        for j in range(len(num_betas)):
            for k in range(len(lattice_sizes)):
                axs['iters'].plot(num_iterations, [vram_usages[(num_ics[i], num_iteration, num_betas[j], lattice_sizes[k])] for num_iteration in num_iterations], '.-')
    axs['iters'].set_title('VRAM usage vs. Number of Iterations')
    axs['iters'].set_xlabel('Number of Iterations')
    axs['iters'].set_ylabel('VRAM Usage (bytes)')
    # axs['iters'].set_xscale('log')
    axs['iters'].set_yscale('log')

    # plot VRAM usage as a function of number of betas
    for i in range(len(num_ics)):
        for j in range(len(num_iterations)):
            for k in range(len(lattice_sizes)):
                axs['pars'].plot(num_betas, [vram_usages[(num_ics[i], num_iterations[j], num_beta, lattice_sizes[k])] for num_beta in num_betas], '.-')
    axs['pars'].set_title('VRAM usage vs. Number of Betas')
    axs['pars'].set_xlabel('Number of Betas')
    axs['pars'].set_ylabel('VRAM Usage (bytes)')
    axs['pars'].set_xscale('log')
    axs['pars'].set_yscale('log')

    # plot VRAM usage as a function of lattice size
    for i in range(len(num_ics)):
        for j in range(len(num_iterations)):
            for k in range(len(num_betas)):
                    axs['size'].plot(lattice_sizes, [vram_usages[(num_ics[i], num_iterations[j], num_betas[k], lattice_size)] for lattice_size in lattice_sizes], '.-')
    axs['size'].set_title('VRAM usage vs. Lattice Size')
    axs['size'].set_xlabel('Lattice Size')
    axs['size'].set_ylabel('VRAM Usage (bytes)')
    # axs['size'].set_xscale('log')
    axs['size'].set_yscale('log')

    plt.tight_layout()
    plt.show()
    
#%%

if __name__ == '__main__':
    vram_usages, (num_ics, num_iterations, num_betas, lattice_sizes) = vram_study()

    plot_study(vram_usages, num_ics, num_iterations, num_betas, lattice_sizes)
    
