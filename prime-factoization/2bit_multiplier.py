import numpy as np
import matplotlib.pyplot as plt
from AND_FA_Hamiltonians import AND, FA

Nbits_in = 2
N_spins = 3 * Nbits_in ** 2 + Nbits_in

J1,h1 = AND(0,Nbits_in,2 * Nbits_in,N_spins)
J_glob = J1.copy()
h_glob = h1.copy()
dummy = 4 * Nbits_in

for i in range(Nbits_in):
    for j in range(Nbits_in):
        j_ = j + Nbits_in
        if not (i == 0 and j == 0):
            J_,h_ = AND(i, j_, dummy, N_spins)
            dummy += 1
            J_glob += J_
            h_glob += h_


J_FA1, h_FA1 = FA(4 * Nbits_in, 4 * Nbits_in + 1, dummy,2 * Nbits_in + 1,dummy+1,N_spins)
J_FA2, h_FA2 = FA(4 * Nbits_in + 2, 4 * Nbits_in + 4, dummy + 1,2 * Nbits_in + 2,2 * Nbits_in + 3,N_spins)

J_glob += J_FA1 + J_FA2
h_glob += h_FA1 + h_FA2

# plt.matshow(J_glob)
# plt.show()

A = [1,1]
B = [1,1]

num_unknowns = 8
all_vecs = []
for i in range(num_unknowns):
    vec = np.zeros(2 ** num_unknowns, dtype = "int") - 1
    for j in range(2 ** i):
        vec[2**i + j::2**(i+1)] = 1
    all_vecs.append(vec)

o0 = all_vecs[0]
o1 = all_vecs[1]
o2 = all_vecs[2]
o3 = all_vecs[3]
w0 = all_vecs[4]
w1 = all_vecs[5]
w2 = all_vecs[6]
w4 = all_vecs[7]

Hamiltonian = []
spin_vecs = []
spin_vec_arr = []
for i,j,k,l,m,n,p,q in zip(o0,o1,o2,o3,w0,w1,w2,w4):
    o = [i, j, k, l]
    w = [m,n,p,-1,q,-1]
    spin_vec = np.array([A[0],A[1],B[0],B[1],o[0],o[1],o[2],o[3],w[0],w[1],w[2],w[3],w[4],w[5]])
    spin_vecs.append(str((l + 1)//2) + str((k + 1)//2) + str((j + 1)//2) + str((i + 1)//2))
    spin_vec_arr.append((spin_vec + 1) // 2)
    energy = 0.5 * np.sum(spin_vec * (J_glob @ spin_vec.transpose())) + np.sum(h_glob * spin_vec)
    Hamiltonian.append(energy)

Hamiltonian = np.array(Hamiltonian)
plt.figure(figsize=[20,8])
plt.bar(spin_vecs,Hamiltonian)

for i in range(len(Hamiltonian)):
    if Hamiltonian[i] == np.max(Hamiltonian):
        print(spin_vecs[i])
        print(spin_vec_arr[i][7],spin_vec_arr[i][6],spin_vec_arr[i][5],spin_vec_arr[i][4])
        print(spin_vec_arr[i][7] * 2**3 + spin_vec_arr[i][6] * 2**2 + spin_vec_arr[i][5] * 2**1 + + spin_vec_arr[i][4])