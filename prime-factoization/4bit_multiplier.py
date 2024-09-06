import numpy as np
import matplotlib.pyplot as plt
from AND_FA_Hamiltonians import AND, FA, generate_4bit_multiplier_Hamiltonian
import random

Nbits_in = 4
N_spins = 3 * Nbits_in ** 2 + Nbits_in
J_glob, h_glob = generate_4bit_multiplier_Hamiltonian(Nbits_in)

# plt.matshow(J_glob)
# plt.show()
print(h_glob)

def Hamiltonian(spin_vec):
    energy = 0.5 * np.sum(spin_vec * (J_glob @ spin_vec)) + np.sum(h_glob * spin_vec)
    return -energy

def simulated_annealing(objective,x,temperature = 25, N_iters = 1000000, inputs = None, outputs = None, n_flips = 1):
    obj_list = []
    curr_x = x.copy()
    best_x = curr_x.copy()
    curr_obj = objective(np.array(x))
    best_obj = curr_obj
    obj_list.append(curr_obj)

    multiply = False
    factor = False

    if inputs is not None:
        multiply = True
        A = inputs[:Nbits_in]
        B = inputs[Nbits_in:]
    if outputs is not None:
        factor = True


    for n in range(N_iters):
        new_candidate = curr_x.copy()
        for i in range(n_flips):
            new_candidate[random.randint(0, N_spins-1)] *= -1
        new_candidate[Nbits_in*4 + Nbits_in * 2 - 1] = -1
        new_candidate[Nbits_in*4 + Nbits_in * 4] = -1
        new_candidate[Nbits_in*4 + Nbits_in * 6] = -1
        new_candidate[Nbits_in*4 + Nbits_in * 8] = -1

        if multiply:
            new_candidate[:Nbits_in] = A
            new_candidate[Nbits_in:Nbits_in * 2] = B
        if factor:
            new_candidate[Nbits_in * 2:Nbits_in * 4] = outputs

        new_obj = objective(np.array(new_candidate))

        if new_obj < curr_obj or random.random() < np.exp((curr_obj - new_obj) / temperature):
            curr_x, curr_obj = new_candidate.copy(), new_obj
            obj_list.append(curr_obj)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = new_candidate.copy()
        if n % 10000 == 0:
            temperature /= 1.1
    return best_x, best_obj, obj_list

# A = [1,1,1,1]
# B = [1,1,1,1]
outputs = [1,1,1,1,-1,-1,-1,1]
import time
begin = time.time()
for n in range(25):
    spin_vec_0 = np.random.randint(0,2,N_spins)
    spin_vec_0[spin_vec_0 == 0] = -1
    spin_vec_0[len(outputs):2 * len(outputs)] = outputs
    # spin_vec_0[:Nbits_in] = A
    # spin_vec_0[Nbits_in:Nbits_in * 2] = B

    x, f, l = simulated_annealing(Hamiltonian,spin_vec_0,outputs=outputs,N_iters=int(1e6))
    # x, f, l = simulated_annealing(Hamiltonian,spin_vec_0,inputs=A+B)

    A_ = (x[:Nbits_in] + 1) // 2
    B_ = (x[Nbits_in:Nbits_in*2] + 1) // 2
    OUT = (x[Nbits_in*2:Nbits_in*4] + 1) // 2
    # print("Factor 1:",np.flip(A_),"Factor 2:",np.flip(B_),"Factored Number:",np.flip(OUT))
    # plt.plot(l)
    # plt.show()

    OUT_ = 0
    for i,o in enumerate(OUT):
        OUT_ += o * 2 ** i
    A__ = 0
    for i,o in enumerate(A_):
        A__ += o * 2 ** i
    B__ = 0
    for i,o in enumerate(B_):
        B__ += o * 2 ** i
    print("Trial no.",n,"Factor 1:",A__, "Factor 2:",B__, "Factored Number:",OUT_,"Min. Hamilt.:", f)

end = time.time()
print(end - begin)

# plt.plot(l)
# plt.show()