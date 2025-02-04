import numpy as np
import matplotlib.pyplot as plt

def plot_energy_convergence(e_history, config, title=None):
    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.subplots_adjust(left=0.05, right=0.825)  # Adjust right to make space for the legend
    if title:
        fig.suptitle(title, horizontalalignment='left', x=0.1)

    ax.set_title('Ising Energy')
    if config.target_energy is not None:
        ax.axhline(config.target_energy, color='k', linestyle=(0, (1, 1)))
    
    # e_history has shape (T, B, I) or (T, 4, B, I) if energies are separated
    other_energies = []
    if e_history.ndim == 4: # I think I removed the separate energies code, so this could be removed too...
        # separate into HP, C1, C2, C3
        e_hp = e_history[:, 0, :, :]
        e_c1 = e_history[:, 1, :, :]
        e_c2 = e_history[:, 2, :, :]
        e_c3 = e_history[:, 3, :, :]
        other_energies = [e_hp, e_c1, e_c2, e_c3]
        total_energy_hist = e_hp + e_c1 + e_c2 + e_c3
    else:
        total_energy_hist = e_history

    num_ics = total_energy_hist.shape[2]
    iters = list(range(total_energy_hist.shape[0]))

    for i, (alpha, beta) in enumerate(config.alpha_beta):
        if num_ics > 1:
            ax.fill_between(iters, total_energy_hist[:, i, :].min(axis=-1), total_energy_hist[:, i, :].max(axis=-1), alpha=0.1, color=f'C{i}')
        # figure out which one has minimum energy at the last iteration
        min_energy_beta_idx = np.argmin(total_energy_hist[-1, i, :])
        ax.plot(iters, total_energy_hist[:, i, min_energy_beta_idx], '.-', lw=0.75, ms=2.0, label=f'α = {alpha:.1e}, β = {beta:.1e}', color=f'C{i}')
        for energy_component, ls in zip(other_energies, (':', '--', '-.', (0, (3, 5, 1, 5, 1, 5)))):
            ax.plot(iters, energy_component[:, i, min_energy_beta_idx], ls=ls, lw=1.0, color=f'C{i}')

    if len(other_energies) > 0:# make custom legend for the energy components
        ax.plot([], [], ls=':', color='k', label='$E_{HP}$')
        ax.plot([], [], ls='--', color='k', label='$E_{1}$')
        ax.plot([], [], ls='-.', color='k', label='$E_{2}$')
        ax.plot([], [], ls=(0, (3, 5, 1, 5, 1, 5)), color='k', label='$E_{3}$')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy')
    
    # move legend outside of the plot
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    
    return fig, ax