import numpy as np
import matplotlib.pyplot as plt

def visualize_spins(spins, cmap='binary', ax=None):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    checkerboard_size = int(np.sqrt(len(spins)))
    ax.imshow(np.reshape(spins, (checkerboard_size, checkerboard_size)), cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax