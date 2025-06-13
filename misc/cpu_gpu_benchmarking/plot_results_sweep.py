# import results fro files bench_cpu_blas_results.npz and bench_cpu_ref_results.npz
# they were saved with np.savez(output_file, deltas=deltas)
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# fit with T=A*n^B
def fit_func(n, A, B, C):
    return A * n**B + C

def load_results(filename):
    """Load benchmark results from a .npz file."""

    # if filename starts with 'results/cbench', then it should end with .bin instead of .npz
    if filename.startswith('results/cbench'):
        filename = filename.replace('.npz', '.bin')
        # if file does not exist, print a warning and return an empty array
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist. Returning empty array.")
            return np.array([])

        # load the binary file
        data = np.fromfile(filename, dtype=np.float64)
        # breakpoint()
        return data
    
    # if file does not exist, print a warning and return an empty array
    if not os.path.exists(filename):
        print(f"Warning: {filename} does not exist. Returning empty array.")
        return np.array([])

    data = np.load(filename)
    return data['deltas']

# plot the results using histograms and twiny 

def plot_results(ref_deltas, blas_deltas):
    """Plot the benchmark results."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ref_deltas_us = ref_deltas * 1e6  # Convert to microseconds
    blas_deltas_us = blas_deltas * 1e6  # Convert to microseconds

    histogram_params = {
        'bins': 50,
        'alpha': 0.75,
        'edgecolor': 'black',
        'linewidth': 0.5,
        'density': False,
    }

    # Plot histogram for reference implementation
    color1 = 'blue'
    main_cluster_max_ref = np.percentile(ref_deltas_us, 99)
    n1, bins1, patches1 = ax1.hist(ref_deltas_us[ref_deltas_us<=main_cluster_max_ref], label='CPU Reference', color=color1, **histogram_params)
    ax1.set_xlabel('Reference Time (μs)')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', labelcolor=color1)

    # Create a second y-axis for the BLAS results
    color2 = 'orange'
    ax2 = ax1.twiny()
    main_cluster_max_blas = np.percentile(blas_deltas_us, 99)
    n2, bins2, patches2 = ax2.hist(blas_deltas_us[blas_deltas_us<=main_cluster_max_blas], label='CPU BLAS', color=color2, **histogram_params)
    ax2.set_xlabel('BLAS Time (μs)', color=color2)
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', labelcolor=color2)

    # Add title and legend
    plt.title('Matmul Benchmark Results: CPU Reference vs BLAS')
    lines = [patches1[0], patches2[0]]
    labels = ['Naive', 'BLAS']
    ax1.legend(lines, labels)

    plt.tight_layout()
    plt.show()

def plot_sweep_results(results):

    colors = [
        "#3d8189",  # blue
        "#efa357",  # orange
        "#18744a",  # green
        "#d2454a",  # red
        "#a476cf",  # purple
        "#7c4a40",  # brown
    ]

    axis_sizes = []

    # make a violin plot of the results
    fig, ax = plt.subplots(figsize=(10, 6))
    for (label, data), col in zip(results.items(), colors):
        sizes = list(data.keys())
        deltas = [data[size] for size in sizes]

        # certain sizes might not have results (will be empty array), so filter them out (and adjust sizes accordingly)
        sizes = [s for s, d in zip(sizes, deltas) if d.size > 0]
        deltas = [d for d in deltas if d.size > 0]
        if not deltas:
            print(f"Warning: No data for {label}. Skipping.")
            continue
        if len(sizes) > len(axis_sizes): # get the largest array to use for axis sizes
            axis_sizes = sizes

        # keep only X-th percentile of the data
        deltas = [d[d != 0.0] for d in deltas] # with no python overhead, some results are bugging out to 0.0, so just removing them for now; better to later change the benchmarking script to take an average over a few runs for each individual point
        deltas = [d[d <= np.percentile(d, 99)] for d in deltas]
        # convert to microseconds
        deltas = [d * 1e6 for d in deltas]
        min_deltas = [np.min(d) for d in deltas]
        max_deltas = [np.max(d) for d in deltas]
        # plot a vertical line from mix to max
        ax.vlines(sizes, min_deltas, max_deltas, color=col)
        # plot the min values as points
        ax.plot(sizes, min_deltas, '-', c=col)
        ax.plot(sizes, min_deltas, 'o', c=col, markersize=9, markeredgecolor="#2a2a2a", label=label)
        
        # # fit the data to the function
        # popt, pcov = curve_fit(fit_func, sizes, min_deltas, p0=(0, 2, 0.2), bounds=([-np.inf, 1.5, 0.1], [np.inf, 2.5, 1]))
        # # generate fitted values
        # fitted_values = fit_func(np.array(sizes), *popt)
        # # plot the fitted line
        # ax.plot(sizes, fitted_values, color=col, linestyle='-', linewidth=1, zorder=0)

        # plot the violin plot
        # parts = ax.violinplot(deltas, positions=sizes, showmeans=False, showmedians=False, showextrema=True, points=1000, widths=[s/10 for s in sizes])
        # for pc in parts['bodies']:
        #     # pc.set_facecolor(col)
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(1)
        # plot the min values as points

    ax.set_xlabel('Matrix Size (n)')
    ax.set_ylabel('Time (µs)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # put the ticks facing inward instead of outward
    ax.tick_params(axis='both', which='major', direction='in')
    ax.tick_params(axis='both', which='minor', direction='in')
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.set_title('MVM Benchmark Results: Digital Electronic vs. Analog Optical Compute')

    # remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set x-ticks to powers of 2 within the range
    min_size = min(axis_sizes)
    max_size = max(axis_sizes)
    xticks = [2**i for i in range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

    ax.legend()



if __name__ == "__main__":

    name_map = {
        'CPU: M2 Max Serial (Naive MVM)': 'bench_cpu_ref',
        'CPU: M2 Max Parallel (Accelerate BLAS)': 'bench_cpu_blas',
        'CPU: M2 Max Parallel (Numpy OpenBLAS)': 'bench_cpu_numpy',
        'GPU: RTX 3060 (Cupy)': 'bench_gpu_rtx3060_cupy',
        'GPU: RTX A6000 (Cupy)': 'bench_gpu_rtxa6000_cupy',
        'CPU: M2 Max Parallel (Accelerate BLAS—No Python)': 'cbench_cpu_blas',
        # 'GPU: RTX 3060 (cuBLAS)': 'bench_gpu_cublas',
        # 'GPU: RTX 3060 (Naive MVM)': 'bench_gpu_ref',
    }

    sizes = [2**i for i in range(4, 14)]
    # sizes = [2**i for i in range(4, 11)]
    results = {
        platform_label: {
            s: load_results(f'data/{filename_prefix}_results_{s}.npz') for s in sizes
        } for platform_label, filename_prefix in name_map.items()
    }

    plot_sweep_results(results)

    ceom_sizes = np.logspace(4, 14, 100, base=2)
    
    ceom_latency_v1 = (1.1526 + (np.array(ceom_sizes)**2/65536*24e-3) + np.array(ceom_sizes)**2 / 106e9) * 1e6
    plt.plot(ceom_sizes, ceom_latency_v1, label='CEOM: Offline DSP + AWG + RTO', color='black', linestyle='-.')
    
    ceom_latency_v2 = ((np.array(ceom_sizes)**2/65536*24e-3) + np.array(ceom_sizes)**2 / 106e9) * 1e6
    plt.plot(ceom_sizes, ceom_latency_v2, label='CEOM: Offline DSP', color='black', linestyle=':')
    
    ceom_latency_v3 = (230e-9 + np.array(ceom_sizes)**2 / 106e9) * 1e6
    plt.plot(ceom_sizes, ceom_latency_v3, label='CEOM: Pipelined DSP', color='black', linestyle='--')
    
    plt.legend()
    plt.show()