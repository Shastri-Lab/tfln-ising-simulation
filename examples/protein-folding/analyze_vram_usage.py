#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
from ising_protein_folding import load_hp_model_by_name, solve_hp_problem_gpu
import matplotlib.cm as cm
import itertools
import datetime
from scipy.optimize import curve_fit

def measure_vram_usage(num_ics, num_iterations, num_betas, lattice_size, model_name=None, skip_energy=False):
    """
    Measure VRAM usage for a given problem size.
    """    
    betas = np.logspace(-4, 0, num_betas)
    model_name = model_name or 'S10'
    model = load_hp_model_by_name(
        model_name,
        latdim=(lattice_size, lattice_size),
        lambdas=(2.1, 2.4, 3.0),
        )
    num_spins = model.len_of_seq * model.dim[0] * model.dim[1] // 2

    torch.cuda.reset_max_memory_allocated()
    solve_hp_problem_gpu(
        model,
        num_iterations=num_iterations,
        num_ics=num_ics,
        alphas=None, 
        betas=betas,
        noise_std=0.3,
        is_plotting=False,
        is_saving=False,
        early_break=False,
        skip_energy=skip_energy,
        )
    vram_usage = torch.cuda.max_memory_allocated()
    
    return vram_usage,num_spins


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
            vram_usage, _ = measure_vram_usage(num_ic, num_iteration, num_beta, lattice_size)
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
    
def vram_spin_study(skip_energy=False):
    """
    Study VRAM usage as a function of number of spins.
    """

    num_ics = [1, 10, 100, 1000]
    num_iterations = [2]
    num_betas = [1]
    lattice_sizes = [4, 5, 6, 7, 8, 9, 10]
    model_names = ['S4', 'S6', 'S7', 'S8', 'S9', 'S10', 'S14', 'S18', 'S20', 'S30', 'S48', 'S64']

    vram_usages = {}
    for num_ic, num_iteration, num_beta, lattice_size, model_name in itertools.product(num_ics, num_iterations, num_betas, lattice_sizes, model_names):
        if num_ic not in vram_usages:
            vram_usages[num_ic] = {}
        try:
            vram_usage, num_spins = measure_vram_usage(num_ic, num_iteration, num_beta, lattice_size, model_name, skip_energy)
        except torch.OutOfMemoryError:
            vram_usage = 12.0 * 2**30 # 12.0 GiB is capacity of 3060 GPU I'm using
        except RuntimeError:
            continue # catch case where lattice size is too small for protein and 
        vram_usages[num_ic][num_spins] = vram_usage

    now = datetime.datetime.now()
    datetime_stamp = now.strftime("%Y%m%d%H%M")
    if not skip_energy:
        filename = f'results/vram_spin_study_{datetime_stamp}.npz'
    else:
        filename = f'results/vram_spin_study_noE_{datetime_stamp}.npz'
    np.savez(filename,
             vram_usages=vram_usages,
             num_ics=num_ics,
             num_iterations=num_iterations,
             num_betas=num_betas,
             lattice_sizes=lattice_sizes,
             )

    return vram_usages, filename

def plot_spin_study(vram_usages, save=True, fn=None):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    num_ics = []
    spins, vrams = [], []
    for n_ic, usage in vram_usages.items():
        num_ics.append(n_ic)
        spins.append([])
        vrams.append([])
        for s, v in usage.items():
            spins[-1].append(s)
            vrams[-1].append(v)

    def exponential(x, a, b, c):
        return a * np.exp(b*x) + c
    
    for i, (spin_list, vram_list, n) in enumerate(zip(spins, vrams, num_ics)):
        
        spins = np.array(spin_list)
        vrams = np.array(vram_list)

        popt, pcov = curve_fit(exponential, spins, vrams, p0=(3e7, 0.0001, 0.0))
        print(f'For {n} ICs:')
        print(f'Params are:\na={popt[0]:.2e}\nb={popt[1]:.2e}\nc={popt[2]:.2e}\n')

        # get logspace domain from min to max of spins
        spins_domain = np.logspace(np.log10(min(spins)), np.log10(max(spins)), 100)
        ax.plot(spins_domain, exponential(spins_domain, *popt), '-', c=f'C{i}', lw=0.5, alpha=0.75)

        ax.plot(spins, vrams, 'x', label=f'{n} ICs', c=f'C{i}')
        # make y position separation linear on the log plot
        y_pos = 3e7 * 10**(0.1*i)
        ax.text(30, y_pos, f'VRAM={popt[0]:.2e} * exp({popt[1]:.2e} * N) + {popt[2]:.2e} bytes', c=f'C{i}')
    
    ax.set_title('VRAM usage vs. Number of Spins')
    ax.set_xlabel('Length of Spin Vector')
    ax.set_ylabel('VRAM Usage (bytes)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    # plt.tight_layout()
    if save:
        if fn is not None:
            plt.savefig(f'{fn}.png')
        else:
            plt.savefig(f'results/vram_spin_study_temp.png')
    else:
        plt.show()

#%%

if __name__ == '__main__':
    # vram_usages, (num_ics, num_iterations, num_betas, lattice_sizes) = vram_study()
    # plot_study(vram_usages, num_ics, num_iterations, num_betas, lattice_sizes)
    
    # plot_spin_study(vram_spin_study(skip_energy=True)[0])

    results, fn = vram_spin_study(skip_energy=False)
    fn = fn.split('.')[0]
    results = np.load(fn+'.npz', allow_pickle=True)
    plot_spin_study(results['vram_usages'][()], save=True, fn=fn)

# %%
