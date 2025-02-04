import itertools
import numpy as np
import matplotlib.pyplot as plt
from matrix_generation import generate_even_sum_set

spin_numbers = [16, 32, 64, 128, 256]
diffs = []

for i,num_spins in enumerate(spin_numbers):
    
    S = generate_even_sum_set(num_spins, 16)

    # generate all possible sequences of ±1 spins
    # sequences = list(itertools.product([0, 1], repeat=num_spins))

    # randomly generate sequences
    num_sequences = 100_000
    sequences = np.random.randint(0, 2, (num_sequences, num_spins))
    spins = sequences * 2 - 1

    # get number based on bits in sequence
    numbers = []
    for seq in sequences:
        seq_str = ''.join(map(str, seq))
        num = int(seq_str, 2)
        numbers.append(num)


    # get energy of each sequence
    energies = np.sum(spins * S, axis=-1) - np.sum(spins * spins, axis=-1) # delta - sum of squares
    
    plt.figure(i+2)
    plt.plot(numbers, energies, 'o', ms=1, c='black', label='data')
    plt.xlabel('Spin configuration as number')
    plt.ylabel('Energy')

    min_e, max_e = energies.min(), energies.max()
    diff = max_e - min_e
    diffs.append(diff)

    # plot energy density
    plt.figure(1)
    plt.hist(energies, bins=50, density=False, alpha=0.6, label=f'N={num_spins}')

plt.xlabel('Energy')
plt.ylabel('Density')
plt.legend()
plt.show()


# plot energy range vs number of spins
plt.figure(2)
plt.plot(spin_numbers, diffs, 'o-', c='black')
plt.xlabel('Number of spins')
plt.ylabel('Energy range')
plt.show()


