#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import networkx as nx
from scipy.io import loadmat

run_analysis = False

mat = loadmat('graphs/g22/G22.mat')
g22_sparse_csc = mat['Problem'][0,0][1]
g22 = g22_sparse_csc.toarray() # g22_sparse_csc is a `scipy.sparse._csc.csc_matrix` object, convert it to a numpy array
G22 = nx.from_numpy_array(g22)
adjacency_matrix = nx.adjacency_matrix(G22).toarray()
num_spins = adjacency_matrix.shape[0]
J = adjacency_matrix

# Function to compute objective value from a spin configuration
def compute_objective(spin_config):
    spins = 2 * spin_config - 1
    energy = spins @ J @ spins
    cut = 0.25*np.sum(J) - 0.25*energy
    return cut


best_g22_cut = 13_359 # this is the best known cut value for this graph

#%%

bit_resolutions = [1,2,3,4,5,6,7,8]

pwd = os.path.dirname(__file__)
files = [f'g22_cut_sweep_{r}bit_2.npy' for r in bit_resolutions]
print('Loading results...')
all_results = {r: np.load(f'{pwd}/results/{file}', allow_pickle=True).item() for r, file in zip(bit_resolutions, files)}


if run_analysis:
    # Dictionary to store final objective values per resolution
    optimal_values = {}

    for r in bit_resolutions:
        print(f'Processing bit resolution {r}...', end=' ')
        results = all_results[r]
        spin_bits_history = np.array(results.spin_bits_history)  # Shape (L,P,M,N)
        energy_history = results.energy_history  # Shape (L,P,M)
        
        # Compute objective values for all iterations, parameters, and trajectories
        print('Computing objective values...', end=' ')
        cut_values = np.apply_along_axis(compute_objective, -1, spin_bits_history)  # Shape (L,P,M)
        
        print('Finding optimal values...', end=' ')
        # Find the iteration where the average objective value is maximized
        avg_cut_values = np.mean(cut_values, axis=(1, 2))  # Shape (L,)
        best_iteration = np.argmax(avg_cut_values)
        
        # Extract optimal objective values at the best iteration
        best_cut_values = cut_values[best_iteration, :, :]  # Shape (P,M)
        
        # Find the best parameter setting that maximizes the objective value
        optimal_param_index = np.argmax(np.mean(best_cut_values, axis=1))
        optimal_values[r] = best_cut_values[optimal_param_index]  # Shape (M,)
        print('Done!')

    # Save optimal values to file
    print('Saving results...', end=' ')
    np.save(f'{pwd}/results/g22_cut_sweep_values_processed_2.npy', optimal_values)
    print('Done!')
else:
    print('Loading processed results...')
    optimal_values = np.load(f'{pwd}/results/g22_cut_sweep_values_processed_2.npy', allow_pickle=True).item()

print('Plotting...')
# Convert to array for plotting
violin_data = [optimal_values[r] for r in bit_resolutions]

# Violin plot of distribution of optimal values per bit resolution
fig = plt.figure(figsize=(6, 4.25))
sns.violinplot(
    data=violin_data,
    inner="point",
    saturation=0.8,
    # color='blue',
    )
plt.xticks(ticks=np.arange(len(bit_resolutions)), labels=bit_resolutions)
plt.xlabel("Bit Resolution")
plt.ylabel("Objective Value Distribution")
# plt.yscale('log')
plt.title("Violin Plot of Optimal Objective Values vs. Bit Resolution")
fig.tight_layout(pad=1.2)
plt.show()


# each element in final_cuts is a list of cut values for each bit resolution
# get the average, std dev, and min/max values for each bit resolution
avg_cuts = np.array([np.mean(optimal_values[res]) for res in optimal_values])
std_cuts = np.array([np.std(optimal_values[res]) for res in optimal_values])
min_cuts = np.array([np.min(optimal_values[res]) for res in optimal_values])
max_cuts = np.array([np.max(optimal_values[res]) for res in optimal_values])

col = 'k'

fig = plt.figure(figsize=(6, 4.25))
plt.plot(bit_resolutions, avg_cuts, 'o', color=col, zorder=3)
# Plot standard deviation as thick vertical lines
for x, avg, std in zip(bit_resolutions, avg_cuts, std_cuts):
    plt.vlines(x, avg - std, avg + std, color=col, linewidth=3, alpha=0.7)

# Plot min/max as thinner vertical lines
for x, min_v, max_v in zip(bit_resolutions, min_cuts, max_cuts):
    plt.vlines(x, min_v, max_v, color=col, linewidth=1.5, alpha=0.5)

# Reference line for best known cut
plt.axhline(best_g22_cut, color='r', linestyle='--', label="Best known cut")
plt.title('G22 cut values: 100 trials w/ T=1,000 ∂=0.95^(t//10)')
plt.xticks(bit_resolutions, bit_resolutions)
plt.xlabel('Bit resolution')
plt.ylabel('Cut value')
plt.grid(True, linestyle='--', alpha=0.6)
fig.tight_layout(pad=1.2)
plt.show()


# Compute fraction of trials exceeding 98% of best_g22_cut
threshold = 0.99 * best_g22_cut
success_fractions = [np.mean(opt_vals >= threshold) for opt_vals in violin_data]

# Plot success fraction as a function of bit resolution
fig = plt.figure(figsize=(6, 4.25))
plt.plot(bit_resolutions, success_fractions, marker='o', linestyle='-')
plt.xlabel("Bit Resolution")
plt.ylabel("Fraction of Trials Above 99% of Best Cut")
plt.title("$TSR_{99}$ vs. Bit Resolution")
plt.grid()
fig.tight_layout(pad=1.2)
plt.show()


