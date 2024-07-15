import os
import json
import numba
import numpy as np
from tqdm import tqdm
import os.path as path
from math import ceil, floor
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from hp_lattice import Lattice_HP_QUBO
from dwave_to_isingmachine import (
    J_dict_to_mat,
    h_dict_to_mat,
)
from ising_protein_folding import load_hp_model_by_name


def compare_energy_functions(model):
    h_dict, J_dict, ising_e_offset = model.to_ising()
    h = h_dict_to_mat(h_dict, model.keys)
    J_sym = J_dict_to_mat(J_dict, model.keys, asymmetric=False)
    J_asym = J_dict_to_mat(J_dict, model.keys, asymmetric=True)

    number_of_spins = len(model.keys)
    number_of_random_configs = 1000

    # generate random QUBO bit strings
    qubo_bits = np.random.randint(0, 2, (number_of_random_configs, number_of_spins))
    # get the corresponding Ising spins
    ising_spins = 2*qubo_bits - 1

    # calculate the QUBO energy using the QUBO dictionary
    qubo_energies = np.zeros(number_of_random_configs)
    for i in range(number_of_random_configs):
        qubo_energies[i] = sum(model.get_energies(qubo_bits[i, :]))

    # calculate the Ising energy using the Ising dictionary
    ising_energies_dict = np.zeros(number_of_random_configs)
    J_dict = defaultdict(float, J_dict)
    for i in range(number_of_random_configs):
        for j1 in range(number_of_spins):
            for j2 in range(number_of_spins):
                ising_energies_dict[i] += J_dict[model.keys[j1], model.keys[j2]] * ising_spins[i, j1] * ising_spins[i, j2]
            ising_energies_dict[i] += h_dict[model.keys[j1]] * ising_spins[i, j1]
        ising_energies_dict[i] += ising_e_offset + model.Lambda[0] * model.len_of_seq

    # calculate the Ising energy using the Ising Hamiltonian
    ising_energies_sym = np.zeros(number_of_random_configs)
    for i in range(number_of_random_configs):
        ising_energies_sym[i] = ising_e_offset + np.dot(ising_spins[i, :], h) + np.dot(ising_spins[i, :], np.dot(J_sym, ising_spins[i, :]))  + model.Lambda[0] * model.len_of_seq
    ising_energies_asym = np.zeros(number_of_random_configs)
    for i in range(number_of_random_configs):
        ising_energies_asym[i] = ising_e_offset + np.dot(ising_spins[i, :], h) + np.dot(ising_spins[i, :], np.dot(J_asym, ising_spins[i, :]))  + model.Lambda[0] * model.len_of_seq

    # assert np.allclose(ising_energies_dict, ising_energies_asym)

    # compare the QUBO and Ising energies
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.plot(qubo_energies, ising_energies_sym, '.', label='Symmetric')
    ax.plot(qubo_energies, ising_energies_asym, '.', label='Asymmetric')
    ax.plot(qubo_energies, ising_energies_dict, '.', label='Dictionary')
    ax.plot(qubo_energies, qubo_energies, 'k--', label='y=x')
    ax.legend()
    ax.set_xlabel('QUBO Energy')
    ax.set_ylabel('Ising Energy')
    ax.set_title(f'QUBO vs Ising Energy for {model.name}')
    plt.show()

def plot_energy_offset():
    models = [
        ('S4', (3,3)),
        ('S6', (3,3)),
        ('S10', (3,4)),
        ('S20', (6,5)),
        ('S30', (7,6)),
        # ('S64', (15,15)),
    ]
    models = [load_hp_model_by_name(name, latdim, lambdas=(2.1, 2.4, 3)) for name, latdim in models]

    delta_es = []
    for model in models:
        h_dict, J_dict, ising_e_offset = model.to_ising()
        h = h_dict_to_mat(h_dict, model.keys)
        J = J_dict_to_mat(J_dict, model.keys, asymmetric=True)
        number_of_spins = len(model.keys)
        qubo_bits = np.random.randint(0, 2, number_of_spins)
        ising_spins = 2*qubo_bits - 1

        qubo_energy = sum(model.get_energies(qubo_bits))
        ising_energy_asym = ising_e_offset + np.dot(ising_spins, h) + np.dot(ising_spins, np.dot(J, ising_spins))
        energy_difference = qubo_energy - ising_energy_asym
        delta_es.append(energy_difference)

    from scipy.optimize import curve_fit
    def linear(x, a, b):
        return a*x + b
    seq_lens = [len(model.sequence) for model in models]
    popt, pcov = curve_fit(linear, seq_lens, delta_es)
    print(f'Energy offset = {popt[0]:.1f}*L + {popt[1]:.2f}')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(seq_lens, delta_es, '.-')
    # put label on each point
    for i, delta_e in enumerate(delta_es):
        ax.text(seq_lens[i], delta_e, f'{models[i].name}', ha='center', va='bottom')

    ax.set_xlabel('Length of Protein Sequence')
    ax.set_ylabel('QUBO - Ising Energy')
    ax.set_title('Energy Offset for Different Models')
    plt.show()

if __name__ == '__main__':
    model = load_hp_model_by_name('S10', (4,3), lambdas=(2.1/6, 3 ,4))
    compare_energy_functions(model)
    # plot_energy_offset()