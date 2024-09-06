import numpy as np
import matplotlib.pyplot as plt
from AND_FA_Hamiltonians import AND, FA, generate_Nbit_multiplier_Hamiltonian,generate_Nbit_sparse_multiplier_Hamiltonian
import random
import time

Ns = np.linspace(4,440,10,dtype = "int")
times = []
dens = []
nnz = []
N_spins = 3 * Ns ** 2 + Ns
for Nbits_in in Ns:
    print(Nbits_in)
    start = time.time()
    J_glob, h_glob = generate_Nbit_sparse_multiplier_Hamiltonian(Nbits_in)
    # h_glob = h_glob[:,np.newaxis]
    end = time.time()
    times.append(end-start)

    dens.append(100 * np.sum(J_glob > 0) / (J_glob.shape[0] * J_glob.shape[1]))
    nnz.append(np.sum(np.abs(J_glob) > 0))
plt.plot(Ns * 2,times)
plt.xlabel("Number of bits to factor")
plt.ylabel("Time to generate Hamiltonian")
plt.show()

plt.plot(Ns * 2,dens)
plt.xlabel("Number of bits to factor")
plt.ylabel("Matrix density (%)")
plt.show()

plt.plot(Ns * 2,nnz)
plt.xlabel("Number of bits to factor")
plt.ylabel("Number of non-zero elements in J")
plt.show()


plt.plot(N_spins,nnz)
plt.xlabel("Number of spins")
plt.ylabel("Number of non-zero elements in J")
plt.tight_layout()
plt.show()

plt.plot(N_spins,np.array(nnz)/N_spins)
plt.xlabel("Number of spins")
plt.ylabel("Average multiplications per spin")
plt.show()