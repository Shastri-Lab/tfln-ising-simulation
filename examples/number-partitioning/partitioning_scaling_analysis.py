#%%
import scipy.io
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from pysing_machine.core.ising_solver import (
    IsingProblem,
    SolverConfig,
    solve_isingmachine,
)
from pysing_machine.core.plotting import plot_energy_convergence
from matrix_generation import generate_even_sum_set, ising_number_partitioning_matrix


#%%


def check_results(spins_bits_history, set_to_partition):
    # spins_bits_history has shape (num_iterations, num_betas, num_ics, num_nodes)

    spins_history = spins_bits_history*2-1
    partition_difference = np.abs((set_to_partition * spins_history).sum(axis=-1))
    zero_indices = np.where(partition_difference == 0, np.arange(partition_difference.shape[0])[:, None, None], np.inf)
    earliest_zero_per_trajectory = np.min(zero_indices, axis=0)

    num_solutions_per_trajectory = np.sum(partition_difference == 0, axis=0)

    trajectory_has_gs = np.any(partition_difference == 0, axis=0)

    probability_of_gs = np.sum(trajectory_has_gs, axis=-1) / spins_history.shape[0]

    breakpoint()

    # breakpoint()

    def spin_vector_to_decimal(spins):
        return int(''.join(['1' if s == 1 else '0' for s in spins]), 2)

    def are_lists_equiv(S1, S2, new_S1, new_S2):
        return sorted([Counter(S1), Counter(S2)]) == sorted([Counter(new_S1), Counter(new_S2)])

    # for each time point, check if we are at ground state. If so, add the result to a list, and then at the end, count the unique solutions in the list.

    # breakpoint()
    num_unique_solutions = np.zeros((spins_history.shape[1], spins_history.shape[2]))
    for i in range(spins_history.shape[1]): # for each beta
        for j in range(spins_history.shape[2]): # for each IC
            # here, we have a trajectory of spins for a single beta and IC
            spins = spins_history[:, i, j, :] # (num_iterations, num_nodes)
            partition_difference = np.abs((set_to_partition * spins).sum(axis=-1)) # (num_iterations,)

            # for each time point, where the partition difference is zero, get the solutoin from the spins
            solutions = spins[np.where(partition_difference == 0)[0], :]
            if solutions.shape[0] == 0:
                continue

            partitions = []
            for k in range(solutions.shape[0]): # loop over the iterations where we have a solution
                set1 = set_to_partition[np.where(solutions[k, :] == 1)[0]]
                set2 = set_to_partition[np.where(solutions[k, :] == -1)[0]]
                if k == 0:
                    partitions.append((set1, set2))
                else:
                    for prev_set1, prev_set2 in partitions:
                        if not are_lists_equiv(set1, set2, prev_set1, prev_set2):
                            partitions.append((set1, set2))
                            break
            
            num_unique_solutions[i, j] = len(partitions)
            
        # breakpoint()
    
    return earliest_zero_per_trajectory, num_unique_solutions, probability_of_gs

def plot_probability_of_gs(results):
    # create a larger figure and set the background color
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)  # adjusted for better spacing

    node_counts = results['node_counts'] # (N,)
    probabilities = np.array(results['probabilities']) # (N, B)

    plt.plot(node_counts, probabilities, '.-', label='Probability of Finding Ground State', color=(83/255,163/255,171/255), linewidth=1)

    plt.xlabel('Number of Nodes', fontsize=14) # , weight='bold')
    plt.ylabel('Probability of Finding Ground State', fontsize=14) # , weight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.xticks(node_counts, fontsize=12, rotation=45)
    # plt.xlim(0, 128)

    return fig, ax

def plot_unique_solutions(results):
    # create a larger figure and set the background color
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)  # adjusted for better spacing

    node_counts = results['node_counts'] # (N,)
    num_unique_solutions = np.array(results['unique_solutions']) # (N, B, ICs)

    colors_for_cmap = [
        (83/255,163/255,171/255), # blue
        (103/255,198/255,95/255), # green
        (228/255,96/255,61/255), # orange
        (230/255,68/255,51/255), # red
    ]

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x']

    for j in range(num_unique_solutions.shape[1]):
        means = []
        medians = []
        stds = []
        mins = []
        maxs = []
        # modes = []
        for i, node_count in enumerate(node_counts):
            num_unique = num_unique_solutions[i, j, :]

            #get non-zero values
            num_unique = num_unique[num_unique != 0]

            means.append(np.mean(num_unique))
            medians.append(np.median(num_unique))
            stds.append(np.std(num_unique))
            mins.append(np.min(num_unique))
            maxs.append(np.max(num_unique))

        ax.plot(node_counts, means, '.-', label=f'Beta {j}', color=colors_for_cmap[j], marker=markers[j % len(markers)], linewidth=2)
        ax.fill_between(node_counts, np.array(mins), np.array(maxs), color=colors_for_cmap[j], alpha=0.2)
        # ax.fill_between(node_counts, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=colors_for_cmap[j], alpha=0.2)

    # axis labels and grid settings
    ax.set_xlabel('Number of Nodes', fontsize=14) # , weight='bold')
    ax.set_ylabel('Number of Unique Solutions Found', fontsize=14) # , weight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # x-axis ticks and labels
    ax.set_xticks(node_counts)
    ax.set_xticklabels(node_counts, fontsize=12, rotation=45)
    # ax.set_xlim(0, 128)

    # legend styling
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
    
    # remove unnecessary plot borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def plot_iterations_histogram(results):
    # create a larger figure and set the background color
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)  # adjusted for better spacing

    node_counts = results['node_counts']  # (N,)
    iterations_to_gs = np.array(results['iterations_to_gs'])  # (N, B, ICs)

    # define color map and marker styles for distinct lines
    colors_for_cmap = [
        (83 / 255, 163 / 255, 171 / 255),  # blue
        (103 / 255, 198 / 255, 95 / 255),  # green
        (228 / 255, 96 / 255, 61 / 255),   # orange
        (230 / 255, 68 / 255, 51 / 255),   # red
    ]

    for i, node_count in enumerate(node_counts):
        iterations = iterations_to_gs[i].flatten()  # flatten to combine all Betas and ICs
        iterations = iterations[(iterations != np.inf) & (iterations < 1000) & (iterations > 0)]  # filter invalid iterations

        if len(iterations) > 0:
            ax.hist(
                iterations, 
                bins=250, 
                alpha=0.5,  # transparency to allow overlap
                label=f'N = {node_count}', 
                color=f'C{i}', 
                edgecolor='black'
            )

    # axis labels and grid settings
    ax.set_xlabel('Iterations to Ground State', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # ax.set_xlim(0, 50)

    # legend styling
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)

    # remove unnecessary plot borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def plot_iterations_to_gs(results):     
    # create a larger figure and set the background color
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)  # adjusted for better spacing

    # title formatting and placement
    # fig.suptitle('Number Partitioning Scaling Analysis', fontsize=18, x=0.5, y=0.95, ha='center', weight='bold')

    # ax.set_title('Iterations to Ground State')
    # ax.set_xlabel('Node Count')
    # ax.set_ylabel('Iterations to Ground State')

    node_counts = results['node_counts'] # (N,)
    iterations_to_gs = np.array(results['iterations_to_gs']) # (N, B, ICs)
    
    # define color map and marker styles for distinct lines
    colors = plt.cm.viridis(np.linspace(0, 1, iterations_to_gs.shape[1]))

    colors_for_cmap = [
        (83/255,163/255,171/255), # blue
        (103/255,198/255,95/255), # green
        (228/255,96/255,61/255), # orange
        (230/255,68/255,51/255), # red
    ]

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x']

    # breakpoint()

    for j in range(iterations_to_gs.shape[1]):
        means = []
        medians = []
        stds = []
        mins = []
        maxs = []
        # modes = []
        for i, node_count in enumerate(node_counts):
            iterations = iterations_to_gs[i, j, :]
            iterations = iterations[(iterations != np.inf) & (iterations < 1000) & (iterations > 0)]

            if len(iterations) == 0:
                # print(f'No iterations for node count {node_count} and beta {j}')
                means.append(np.nan) # means.append(0)
                medians.append(np.nan) # medians.append(0) 
                stds.append(0)
                mins.append(0)
                maxs.append(0)
            else:
                means.append(np.mean(iterations))
                medians.append(np.median(iterations))
                stds.append(np.std(iterations))
                mins.append(np.min(iterations))
                maxs.append(np.max(iterations))
            
            # modes.append(np.argmax(np.bincount(iterations)))

            # violin_parts = ax.violinplot(iterations, positions=[node_count], widths=10, showmeans=True, showextrema=False, showmedians=False)
            # for pc in violin_parts['bodies']:
            #     pc.set_facecolor(f'C{j}')
            #     pc.set_edgecolor('black')
            #     pc.set_alpha(0.15)

        ax.plot(node_counts, means, label=f'Beta {j}', color=colors_for_cmap[j], marker=markers[j % len(markers)], linewidth=2)
        ax.fill_between(node_counts, np.array(mins), np.array(maxs), color=colors_for_cmap[j], alpha=0.2)
        # ax.fill_between(node_counts, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=colors_for_cmap[j], alpha=0.2)

        # axs[1].plot(node_counts, medians, '.-', label=f'Beta {j}', color=f'C{j}')
        # axs[2].plot(node_counts, modes, '.', label=f'Beta {j}', color=f'C{j}')

    # axis labels and grid settings
    ax.set_xlabel('Number of Nodes', fontsize=14) # , weight='bold')
    ax.set_ylabel('Mean Iterations to Ground State', fontsize=14) # , weight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # x-axis ticks and labels
    ax.set_xticks(node_counts)
    ax.set_xticklabels(node_counts, fontsize=12, rotation=45)
    # ax.set_xlim(0, 512)

    # legend styling
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
    
    # remove unnecessary plot borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

#%%
def run_scaling_analysis_mat():

    node_counts = (16, 32, 64, 128) # , 256
    filenames = (f'mat_files/numberPartitioning_{n}N.mat' for n in node_counts)
    mat_files = (scipy.io.loadmat(f) for f in filenames)

    iterations_to_gs = []
    unique_solutions = []
    probabilities = []

    for mat in mat_files:
        J = mat['J']
        h = np.zeros(J.shape[0])
        set_to_partition = mat['nset'][0]

        problem = IsingProblem(J=J, h=h)
        config = SolverConfig(
            target_energy=None, # -(nset**2).sum(),
            num_iterations=1000,
            num_ics=5_000,
            alphas=None,
            betas=0.01,
            noise_std=0.1,
            simulated_annealing=False,
            ic_range=0.01,
        )

        results = solve_isingmachine(problem, config)

        spins_bits_history = np.array(results.spin_bits_history)
        earliest_zero, num_unique_solutions, probability_of_gs = check_results(spins_bits_history, set_to_partition)
        iterations_to_gs.append(earliest_zero)
        unique_solutions.append(num_unique_solutions)
        probabilities.append(probability_of_gs)

    analysis_results = {
        'node_counts': node_counts,
        'iterations_to_gs': iterations_to_gs, # for each trajectory, index of the first time the ground state is reached
        'unique_solutions': unique_solutions,
        'probabilities': probabilities,
    }
    np.save('results/number_partitioning_results_test_mat.npy', analysis_results)

    return analysis_results

def run_scaling_analysis():

    # node_counts = np.linspace(16, 1024, 10)
    node_counts = [
        8, 16, 32, 64, 128
    ]

    iterations_to_gs = []
    unique_solutions = []
    probabilities = []

    for N  in node_counts:
        set_to_partition = generate_even_sum_set(int(N), 16)
        J = ising_number_partitioning_matrix(set_to_partition)
        h = np.zeros(J.shape[0])

        problem = IsingProblem(J=J, h=h)
        config = SolverConfig(
            target_energy=None, # -(nset**2).sum(),
            num_iterations=1000,
            num_ics=10_000,
            alphas=None,
            betas=0.03,
            noise_std=0.15,
            ic_range=0.05,
            simulated_annealing=False,
            make_symmetric=False,
        )

        results = solve_isingmachine(problem, config)

        # save results to a file
        # np.save(f'results/number_partitioning_results_{N}.npy', results)

        # breakpoint()
        spins_bits_history = np.array(results.spin_bits_history)
        earliest_zero, num_unique_solutions, probability_of_gs = check_results(spins_bits_history, set_to_partition)
        iterations_to_gs.append(earliest_zero)
        unique_solutions.append(num_unique_solutions)
        probabilities.append(probability_of_gs)


    analysis_results = {
        'node_counts': node_counts,
        'iterations_to_gs': iterations_to_gs, # for each trajectory, index of the first time the ground state is reached
        'unique_solutions': unique_solutions,
        'probabilities': probabilities,
    }
    np.save('results/number_partitioning_results_test1.npy', analysis_results)

    return analysis_results


def plot_saved_analysis(filename):
    results = np.load(filename, allow_pickle=True).item()
    if 'iterations_to_gs' in results:
        plot_iterations_to_gs(results)
        plot_iterations_histogram(results)
    if 'unique_solutions' in results:
        plot_unique_solutions(results)
    if 'probabilities' in results:
        plot_probability_of_gs(results)
    plt.show()

#%%

if __name__ == '__main__':
    # results = run_scaling_analysis()
    # plot_probability_of_gs(results)
    # plt.show()
    # plot_iterations_to_gs(results)
    # plt.show()
    # plot_iterations_histogram(results)
    # plt.show()
    # plot_unique_solutions(results)
    # plt.show()
    # plot_saved_analysis('results/number_partitioning_results_test_mat.npy')
    plot_saved_analysis('results/number_partitioning_results_test1.npy')
    # plot_saved_analysis('results/number_partitioning_results_experimental.npy')
    # plot_saved_analysis('results/number_partitioning_results_3.npy')

# %%
