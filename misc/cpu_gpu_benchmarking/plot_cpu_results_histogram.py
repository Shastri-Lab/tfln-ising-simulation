# import results fro files bench_cpu_blas_results.npz and bench_cpu_ref_results.npz
# they were saved with np.savez(output_file, deltas=deltas)
import numpy as np
import matplotlib.pyplot as plt

def load_results(filename):
    """Load benchmark results from a .npz file."""
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

if __name__ == "__main__":
    # Load results from files
    ref_deltas = load_results('data/bench_cpu_ref_results.npz')
    blas_deltas = load_results('data/bench_cpu_blas_results.npz')

    # Plot the results
    plot_results(ref_deltas, blas_deltas)