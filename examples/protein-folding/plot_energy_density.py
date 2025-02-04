import itertools
import numpy as np
from ising_protein_folding import load_hp_model_by_name
import matplotlib.pyplot as plt

configurations = [
    # ("S4", (2,2)),
    ("S10", (3,4)),
]

models = [
    load_hp_model_by_name(
        name,
        latdim=latdim,
        lambdas=(2.1, 2.4, 3.0),
        )
    for name, latdim in configurations
]

for model in models:
    num_spins = len(model.keys)
    
    # generate all possible sequences of ±1 spins
    # sequences = list(itertools.product([0, 1], repeat=num_spins))

    # randomly generate sequences
    num_sequences = 100_000
    sequences = np.random.randint(0, 2, (num_sequences, num_spins))

    # get energy of each sequence
    energies = [sum(model.get_energies(seq)) for seq in sequences]

    # for s, e in zip(sequences, energies):
    #     print(f'{s} -> {e}')

    # plot energy density
    plt.hist(energies, bins=50, density=False)
    plt.title(f'Energy Density for {model.name}')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.show()


