#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import networkx as nx
from scipy.io import loadmat
import pandas

run_analysis = False
is_saving = False

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

    if is_saving:
        # Save optimal values to file
        print('Saving results...', end=' ')
        np.save(f'{pwd}/results/g22_cut_sweep_values_processed_2.npy', optimal_values)
        print('Done!')
else: # load
    print('Loading processed results...')
    optimal_values = np.load(f'{pwd}/results/g22_cut_sweep_values_processed_2.npy', allow_pickle=True).item()


#%%
# plotting optimal value distributions as a violin plot
violin_data = [100 * optimal_values[r] / best_g22_cut for r in bit_resolutions]
violin_data = pandas.DataFrame(violin_data).T
violin_data.columns = bit_resolutions
violin_data = violin_data.melt(var_name='Bit Resolution', value_name='Percentage of Best Known Cut')

col1 = (110/255,136/255,194/255, 255/255) # 'darkcyan'
col2 = (108/255,101/255,170/255,255/255)

fig = plt.figure(figsize=((8.5-2)/2*1.2, (8.5-2)/2*3/4*1.2))
# fig = plt.figure(figsize=(4, 3))
sns.violinplot(
    x='Bit Resolution',
    y='Percentage of Best Known Cut',
    data=violin_data,
    inner="point",
    color=col1,
    linecolor=(0.1,0.1,0.1,0.8),
    density_norm='count',
    )
baseax = plt.gca()

exp_best_result = (3.3-1, 99.7)
baseax.plot(exp_best_result[0], exp_best_result[1], 'o', markerfacecolor=col2, markeredgecolor='darkslateblue')

plt.ylim(0, 103)
baseax.spines['left'].set_bounds(0, 100) # Modify the left spine to end at 100
sns.despine() # remove top and left borders
baseax.yaxis.set_major_locator(plt.MultipleLocator(20))
baseax.yaxis.set_minor_locator(plt.MultipleLocator(10))
baseax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}' if x % 20 == 0 else ''))
plt.tick_params(axis='both', which='both', direction='in')
plt.xlabel("Bit Resolution")
plt.ylabel("Proximity to Best Cut (%)")
# plt.title("Simulated Ising Machine G22 Max-Cut")


# zoomed in inset for distributions cramped into the top
inset_x_min = 2.8-1
inset_x_max = 7.47
inset_y_min = 95.5
inset_y_max = 100.2
# axins = fig.add_axes([0.5, 0.22, 0.4, 0.5])
axins = baseax.inset_axes(
    [0.45, 0.15, 0.5, 0.55],
    xlim=(inset_x_min, inset_x_max), ylim=(inset_y_min, inset_y_max)) #, xticklabels=[])#, yticklabels=[])

sns.violinplot(
    x='Bit Resolution',
    y='Percentage of Best Known Cut',
    data=violin_data,
    inner="point",
    color=col1,
    linecolor=(0.1,0.1,0.1,0.8),
    density_norm='count',
    ax=axins
    )
axins.set_xlabel('') # remove labels automatically added to the inset by violin plot
axins.set_ylabel('')
axins.yaxis.set_major_locator(plt.MultipleLocator(1))
axins.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
axins.tick_params(axis='both', which='both', direction='in')

axins.plot(exp_best_result[0], exp_best_result[1], 'o', markerfacecolor=col2, markeredgecolor='darkslateblue')
# axins.annotate('Best Exp.', xy=exp_best_result, xytext=(exp_best_result[0] + 1.2, exp_best_result[1] + 0.2), arrowprops=dict(facecolor='darkslateblue', shrink=0.05, width=2, headwidth=6, headlength=5))
text_pos = (4.6, 80)
baseax.text(*text_pos, 'Analog Optical\n(This Work)', c=col2, fontsize=8, va='center', ha='left', zorder=10)
# draw an arrow from the text to the point
arrow_dict = dict(facecolor=col2, edgecolor=(0,0,0,0), shrink=0.025, width=1.5, headwidth=4, headlength=5)
baseax.annotate('', xy=exp_best_result, xytext=text_pos, arrowprops=arrow_dict, fontsize=8)
baseax.annotate('', xy=(3.54, 66.5), xytext=text_pos, arrowprops=arrow_dict, fontsize=8, zorder=10)

axins.text(3.6, 97.5, 'Digital Simulation', c=col1, fontsize=8, va='center', ha='left')

axins.set_xlim(inset_x_min, inset_x_max)
axins.set_ylim(inset_y_min, inset_y_max)


# put a box around the area in the main plot that is shown in the inset
rect = plt.Rectangle((inset_x_min, inset_y_min), inset_x_max - inset_x_min, inset_y_max - inset_y_min, edgecolor='k', facecolor='none', lw=0.5)
baseax.add_patch(rect)
# baseax.indicate_inset_zoom(axins, edgecolor="black", alpha=1, linewidth=0.5, hatch_linewidth=0.5)

px, py = axins.transData.transform((inset_x_min, inset_y_max))
x_main, y_main = baseax.transData.inverted().transform((px, py))
baseax.plot([x_main, inset_x_min], [y_main, inset_y_min], '-', c='k', lw=0.5)

px, py = axins.transData.transform((inset_x_max, inset_y_max))
x_main, y_main = baseax.transData.inverted().transform((px, py))
baseax.plot([x_main, inset_x_max], [y_main, inset_y_min], '-', c='k', lw=0.5)

# plt.grid()
fig.tight_layout(pad=0.5)
plt.savefig(f'{pwd}/results/g22_cut_sweep_violin_plot_2.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%

fig = plt.figure(figsize=(6/6*4, 4.25/6*4))

fractions_above_98 = [(optimal_values[r]/best_g22_cut > .98).mean() for r in bit_resolutions]
fractions_above_99 = [(optimal_values[r]/best_g22_cut > .99).mean() for r in bit_resolutions]
fractions_above_99_5 = [(optimal_values[r]/best_g22_cut > .995).mean() for r in bit_resolutions]

c1 = (110/255,136/255,194/255, 255/255)
c2 = (218/255,43/255,129/255,255/255)
c3 = (108/255,101/255,170/255,255/255)

ax = plt.gca()
ax.set_axisbelow(True) 
# Plot the fractions as a bar plot
plt.plot(bit_resolutions, fractions_above_98, '-o', label=r'$\geq$ 98%',c=c1, zorder=10)
plt.plot(bit_resolutions, fractions_above_99, '-o', label=r'$\geq$ 99%',c=c2, zorder=10)
plt.plot(bit_resolutions, fractions_above_99_5, '-o', label=r'$\geq$ 99.5%',c=c3, zorder=10)
plt.xlabel("Bit Resolution")
plt.ylabel("Quantile-conditioned success rate")
# plt.title("Fraction of Data Points Above 99% for Each Bit Resolution")
plt.ylim(-0.1, 1.1)
plt.legend(loc='upper left')
# plt.grid()
plt.tight_layout(pad=0.5)
plt.savefig(f'{pwd}/results/g22_cut_sweep_fractions_2.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
print("Exporting plot data to Excel...")

# Create a Pandas Excel writer using XlsxWriter as the engine
excel_path = os.path.join(pwd, 'g22_cut_plot_data.xlsx')
with pandas.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    # Write violin plot data
    violin_data_wide = pandas.DataFrame({r: 100 * optimal_values[r] / best_g22_cut for r in bit_resolutions})
    violin_data_wide.to_excel(writer, sheet_name='Violin Data', index=False)

    # Write fraction-above-thresholds data
    thresholds_df = pandas.DataFrame({
        'Bit Resolution': bit_resolutions,
        '>=98%': fractions_above_98,
        '>=99%': fractions_above_99,
        '>=99.5%': fractions_above_99_5
    })
    thresholds_df.to_excel(writer, sheet_name='Threshold Fractions', index=False)

print(f"Data exported to {excel_path}")

# %%
