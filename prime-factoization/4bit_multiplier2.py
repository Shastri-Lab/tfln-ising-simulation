import numpy as np
import matplotlib.pyplot as plt
from AND_FA_Hamiltonians import AND, FA
import random
from Hamiltonian import construct_J,construct_h

Nbits_in  = 3
J_glob = construct_J(Nbits_in)
h_glob = construct_h(Nbits_in)

plt.matshow(J_glob)
plt.show()
print(h_glob)
def Hamiltonian(spin_vec):
    energy = 0.5 * np.sum(spin_vec * (J_glob @ spin_vec.transpose())) + np.sum(h_glob * spin_vec)
    return -energy

def simulated_annealing(objective,x,temperature = 100, N_iters = 1000000):
    obj_list = []
    best_x = x
    curr_obj = objective(np.array(x))
    best_obj = curr_obj
    obj_list.append(curr_obj)
    for n in range(N_iters):
        # for j in range(x.shape[0] - Nbits_in * 2):
        new_candidate = x.copy()
        new_candidate[random.randint(0, N_spins - 4 - 1)] *= -1
        # new_candidate[j] *= -1
        new_obj = objective(new_candidate)
        if new_obj < curr_obj or random.random() < np.exp((curr_obj - new_obj) / temperature):
            obj_list.append(new_obj)
            x, curr_obj = new_candidate, new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = x.copy()
        if n % 10000 == 0:
            temperature /= 1.5
    return best_x, best_obj, obj_list

N_spins = 3 * Nbits_in ** 2 #+ Nbits_in

outputs = np.array([1,-1,-1,1]) * 1
# outputs *= -10
# output = [1,0,0,0,1,1,1,1]

for i in range(15):
    spin_vec_0 = np.random.randint(0,2,N_spins)
    spin_vec_0[spin_vec_0 == 0] = -1
    spin_vec_0[-2*Nbits_in:] = outputs

    x, f, l = simulated_annealing(Hamiltonian,spin_vec_0)

    A = (1 * x[:Nbits_in] + 1) / 2
    B = (1 * x[Nbits_in:2*Nbits_in] + 1) / 2
    Out = (1*x[-2*Nbits_in:] + 1) / 2
    print(A,B,Out)
