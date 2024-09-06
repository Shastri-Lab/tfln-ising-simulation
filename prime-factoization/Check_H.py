import numpy as np
import matplotlib.pyplot as plt
from AND_FA_Hamiltonians import AND, FA, generate_Nbit_multiplier_Hamiltonian,generate_Nbit_multiplier_Hamiltonian_old
import random
import time


Nbits_in = 9
J_glob, h_glob = generate_Nbit_multiplier_Hamiltonian(Nbits_in)
J_glob_old, h_glob_old = generate_Nbit_multiplier_Hamiltonian_old(Nbits_in)

print(np.sum(J_glob - J_glob_old))
print(np.sum(h_glob - h_glob_old))
