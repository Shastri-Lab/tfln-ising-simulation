# open the image in scaled_black_cat.png

import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path and convert it to a numpy array.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        np.ndarray: The image as a numpy array.
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

image = load_image("images/scaled_black_cat.png")

image_as_spins = (image > 128).astype(np.int8) * 2 - 1  # Convert to Ising spins (-1, 1)

image_spins_array = image_as_spins.reshape(-1)

# turn into a hopfield matrix
import numpy as np

def create_hopfield_matrix_from_spins(memories):
    # use outer product to create the Hopfield matrix
    N = len(memories)
    n = len(memories[0])
    hopfield_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                for spins in memories:
                    hopfield_matrix[i, j] += spins[i] * spins[j]
    hopfield_matrix /= N  # Normalize by the number of memories
    return hopfield_matrix


def create_kings_graph_matrix_from_spins(spins, grid_size=(20, 20)):
    n = len(spins)
    kings_graph_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the row and column indices in the non-square grid
                row_i, col_i = divmod(i, grid_size[1])
                row_j, col_j = divmod(j, grid_size[1])
                
                # Check if they are neighbors in a kings graph (8 directions)
                if (abs(row_i - row_j) <= 1 and abs(col_i - col_j) <= 1):
                    kings_graph_matrix[i, j] = spins[i] * spins[j]

    return kings_graph_matrix

hopfield_matrix = create_kings_graph_matrix_from_spins(image_spins_array, grid_size=image_as_spins.shape)
# hopfield_matrix = create_hopfield_matrix_from_spins([image_spins_array])

from pysing_machine.core.ising_solver import (
    IsingProblem,
    SolverConfig,
    solve_isingmachine,
)

noise_std = 0.5  # Standard deviation of noise for the annealing schedule
def annealing_schedule(t):
    return noise_std * 0.95**(t//10)

max_epsilon = np.min(np.abs(hopfield_matrix))*3
epsilon = max_epsilon / 4
h = -epsilon*image_spins_array
ising_problem = IsingProblem(J=hopfield_matrix, h=h)
config = SolverConfig(
    # target_energy=min_energy,
    num_iterations=1000,
    num_ics=50,
    alphas=(1,),
    betas=(0.7, 0.5,),
    start_temperature=noise_std,
    annealing_schedule='exponential',
    early_break=False,
    sparse=False,
)

results = solve_isingmachine(ising_problem, config)

final = results.final_vector # has shape (1, 10, 400), need to make into 10 different 20x20 images
final_images = final.reshape(-1, 20, 20)
import matplotlib.pyplot as plt
for i, img in enumerate(final_images):
    # convert to 1/0
    img = (img > 0).astype(np.uint8) * 255  # Convert to binary image (0 or 255)
    plt.imshow(img, cmap='gray')
    plt.title(f'Image {i+1}')
    plt.axis('off')
    plt.show()