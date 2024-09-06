import numpy as np
import matplotlib.pyplot as plt
from AND_FA_Hamiltonians import AND, FA, generate_Nbit_multiplier_Hamiltonian, generate_Nbit_sparse_multiplier_Hamiltonian
import random
import time
Nbits_in = 8
N_spins = 3 * Nbits_in ** 2 + Nbits_in

start = time.time()
J_glob, h_glob = generate_Nbit_multiplier_Hamiltonian(Nbits_in)
h_glob = h_glob[:,0]
J_glob_s, h_glob_s = generate_Nbit_sparse_multiplier_Hamiltonian(Nbits_in)
print(J_glob_s.nnz / (J_glob_s.shape[0] * J_glob_s.shape[1]), J_glob_s.shape[0])
h_glob = h_glob[:,np.newaxis]
end = time.time()
print(end - start)
# plt.matshow(J_glob)
# plt.show()
# print(h_glob)
# J_glob_s = J_glob_s.tobsr()
# h_glob_s = h_glob_s.tobsr()

# def Hamiltonian(spin_vec):
#     a = J_glob @ spin_vec.transpose()
#     b = J_glob @ spin_vec[0,:]
#     print(a[:,0],b, a[:,0] - b)
#     print(a.shape, spin_vec.shape, (h_glob * spin_vec).shape)
#     energy = 0.5 * np.sum(spin_vec.transpose() * (J_glob @ spin_vec.transpose()),axis = 0) + np.sum(h_glob * spin_vec,axis = 1)
#     return -energy

# def Hamiltonian(spin_vec):
#     energy = 0.5 * np.sum(spin_vec.transpose() * (J_glob @ spin_vec.transpose()),axis = 0) + np.sum(h_glob * spin_vec,axis = 1)
#     print(energy)
#     energy2 = 0.5 * np.sum(spin_vec[0,:] * (J_glob @ spin_vec[0,:])) + np.sum(h_glob * spin_vec[0,:])
#     print(energy2)
#     return -energy

# def Hamiltonian(spin_vec):
#     # sv = spin_vec[0,:]
#     # energy0 = 0.5 * np.sum(sv * (J_glob @ sv)) + np.sum(h_glob * sv)
#     # print(J_glob.shape,spin_vec.shape)
#     energy = 0.5 * np.sum(spin_vec * (J_glob @ spin_vec),axis = 0) + np.sum(h_glob * spin_vec,axis = 0)
#     return -energy

def Hamiltonian(spin_vec):
    # begin = time.time()
    # energy = 0.5 * np.sum(spin_vec * (J_glob_s @ spin_vec),axis = 0) + np.sum(h_glob_s * spin_vec,axis = 0)
    # end = time.time()
    # begin1 = time.time()
    energy0 = 0.5 * np.sum(spin_vec * (J_glob @ spin_vec),axis = 0) + np.sum(h_glob * spin_vec,axis = 0)
    # end1 = time.time()
    # print(end1 - begin1, end - begin)
    # if np.any(energy - energy0):
    #   print(energy - energy0)
    return -energy0


def Interaction(spin_vec):
    return  J_glob @ spin_vec + h_glob
def Interaction_Gibbs(spin_vec,i):
    # print((np.sum(J_glob[i,:,np.newaxis] * spin_vec,axis = 0)+ h_glob[i]).shape)
    return np.sum(J_glob[i,:,np.newaxis] * spin_vec, axis = 0) + h_glob[i]
def pbit_method_Gibbs(objective,x,temperature = 25, N_iters = 1000000, inputs = None,
                        outputs = None,  N_updates_btw_anneal = 10000):

    curr_x = x.copy()
    obj_list = []
    curr_x[Nbits_in * 4 + Nbits_in * 2 - 1, :] = -1
    for i in range(Nbits_in - 1):
        curr_x[Nbits_in * 4 + Nbits_in ** 2 + Nbits_in * (2 * i), :] = -1
    best_x = curr_x.copy()
    curr_obj = objective(curr_x)
    best_obj = curr_obj.copy()
    obj_list.append(curr_obj.copy())
    print(best_obj)
    multiply = False
    factor = False

    if inputs is not None:
        multiply = True
        A = inputs[:Nbits_in]
        B = inputs[Nbits_in:]
    if outputs is not None:
        factor = True

    # beta = 4.0
    for n in range(N_iters):
        if n%1000==0:
            print(n)
        for m in range(h_glob.shape[0]):
            new_candidate = curr_x.copy()
            new_candidate[m,:] = np.sign(np.tanh(1.0/temperature * Interaction_Gibbs(new_candidate,m)) + np.random.uniform(-1,1, new_candidate[m,:].shape))
            new_candidate[Nbits_in*4 + Nbits_in * 2 - 1,:] = -1
            for i in range(Nbits_in - 1):
                new_candidate[Nbits_in*4 + Nbits_in ** 2 + Nbits_in * (2*i),:] = -1

            if multiply:
                new_candidate[:Nbits_in,:] = A
                new_candidate[Nbits_in:Nbits_in * 2,:] = B
            if factor:
                new_candidate[Nbits_in * 2:Nbits_in * 4,:] = outputs


            curr_x = new_candidate.copy()
        new_obj = objective(new_candidate)
        curr_obj = new_obj.copy()
        obj_list.append(curr_obj.copy())
        best_update = new_obj < best_obj
        best_obj[best_update] = new_obj[best_update].copy()
        best_x[:,best_update] = new_candidate[:,best_update].copy()
        if n % N_updates_btw_anneal == 0:
            temperature /= 1.1
    return best_x, best_obj, obj_list


def pbit_method(objective,x,temperature = 25, N_iters = 1000000, inputs = None,
                        outputs = None,  N_updates_btw_anneal = 10000):

    curr_x = x.copy()
    obj_list = []
    curr_x[Nbits_in * 4 + Nbits_in * 2 - 1, :] = -1
    for i in range(Nbits_in - 1):
        curr_x[Nbits_in * 4 + Nbits_in ** 2 + Nbits_in * (2 * i), :] = -1
    best_x = curr_x.copy()
    curr_obj = objective(curr_x)
    best_obj = curr_obj.copy()
    obj_list.append(curr_obj.copy())
    print(best_obj)
    multiply = False
    factor = False

    if inputs is not None:
        multiply = True
        A = inputs[:Nbits_in]
        B = inputs[Nbits_in:]
    if outputs is not None:
        factor = True
    for n in range(N_iters):
        new_candidate = np.sign(np.tanh(1.0/temperature * Interaction(curr_x)) + np.random.uniform(-1,1,curr_x.shape))
        new_candidate[Nbits_in*4 + Nbits_in * 2 - 1,:] = -1
        for i in range(Nbits_in - 1):
            new_candidate[Nbits_in*4 + Nbits_in ** 2 + Nbits_in * (2*i),:] = -1

        if multiply:
            new_candidate[:Nbits_in,:] = A
            new_candidate[Nbits_in:Nbits_in * 2,:] = B
        if factor:
            new_candidate[Nbits_in * 2:Nbits_in * 4,:] = outputs

        new_obj = objective(new_candidate)

        # update = (new_obj < curr_obj) | (np.random.random(new_obj.shape) < np.exp((curr_obj - new_obj) / temperature))
        update = True
        curr_x[:,update] = new_candidate[:,update].copy()
        curr_obj[update] = new_obj[update].copy()
        obj_list.append(curr_obj.copy())
        best_update = new_obj < best_obj
        best_obj[best_update] = new_obj[best_update].copy()
        best_x[:,best_update] = new_candidate[:,best_update].copy()
        if n % N_updates_btw_anneal == 0:
            temperature /= 1.1
    return best_x, best_obj, obj_list

def simulated_annealing(objective,x,temperature = 25, N_iters = 1000000, inputs = None,
                        outputs = None, n_flips = 1, N_updates_btw_anneal = 10000):
    obj_list = []
    curr_x = x.copy()
    curr_x[Nbits_in * 4 + Nbits_in * 2 - 1,:] = -1
    for i in range(Nbits_in - 1):
        curr_x[Nbits_in * 4 + Nbits_in ** 2 + Nbits_in * (2 * i),:] = -1
    best_x = curr_x.copy()
    curr_obj = objective(x)
    best_obj = curr_obj.copy()
    obj_list.append(curr_obj)

    multiply = False
    factor = False

    if inputs is not None:
        multiply = True
        A = inputs[:,Nbits_in]
        B = inputs[:,Nbits_in:]
    if outputs is not None:
        factor = True

    for n in range(N_iters):
        new_candidate = curr_x.copy()
        for i in range(n_flips):
            new_candidate[np.random.randint(0, N_spins-1),:] *= -1

        new_candidate[Nbits_in*4 + Nbits_in * 2 - 1,:] = -1
        for i in range(Nbits_in - 1):
            new_candidate[Nbits_in*4 + Nbits_in ** 2 + Nbits_in * (2*i),:] = -1

        if multiply:
            new_candidate[:Nbits_in,:] = A
            new_candidate[Nbits_in:Nbits_in * 2,:] = B
        if factor:
            new_candidate[Nbits_in * 2:Nbits_in * 4,:] = outputs

        new_obj = objective(new_candidate)

        # update = (new_obj < curr_obj) | (np.random.random(curr_obj.shape[0]) < np.exp((curr_obj - new_obj) / temperature))
        update = (new_obj < curr_obj) | (np.random.random(new_obj.shape) < np.exp((curr_obj - new_obj) / temperature))
        curr_x[:,update] = new_candidate[:,update].copy()
        curr_obj[update] = new_obj[update].copy()
        obj_list.append(curr_obj.copy())
        best_update = new_obj < best_obj
        best_obj[best_update] = new_obj[best_update].copy()
        best_x[:,best_update] = new_candidate[:,best_update].copy()
        if n % N_updates_btw_anneal == 0:
            temperature /= 1.1
    return best_x, best_obj, obj_list
# A = [1,1,1,1]
# B = [1,1,1,1]
N_ics = 100
# outputs = np.array([1,1,1,1,-1,-1,-1,1])[:,np.newaxis]
# outputs = np.array([1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1])[:,np.newaxis]
outputs = np.array([1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,1,-1,1])[:,np.newaxis]


OUT_ = 0
for i, o in enumerate((outputs + 1) // 2):
    OUT_ += o * 2 ** i
print(OUT_)

begin = time.time()
for n in range(1):
    spin_vec_0 = np.random.randint(0,2,(N_spins,N_ics))
    spin_vec_0[spin_vec_0 == 0] = - 1
    spin_vec_0[len(outputs):2 * len(outputs),:] = outputs
    # spin_vec_0[:Nbits_in] = A
    # spin_vec_0[Nbits_in:Nbits_in * 2] = B

    # x, f, l = simulated_annealing(Hamiltonian,outputs=outputs, x = spin_vec_0, N_iters=int(5e4), N_updates_btw_anneal = 1000, temperature=10)
    x, f, l = pbit_method_Gibbs(Hamiltonian,outputs=outputs, x = spin_vec_0, N_iters=int(20e4), N_updates_btw_anneal = 2500, temperature=10)

    print(f.transpose(), Hamiltonian(x).transpose())
    A_ = (x[:Nbits_in] + 1) // 2
    B_ = (x[Nbits_in:Nbits_in*2] + 1) // 2
    OUT = (x[Nbits_in*2:Nbits_in*4] + 1) // 2

    print(A_.transpose())
    print(B_.transpose())
    print(OUT.transpose())
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
    print("F1:",A__)
    print("F2 :",B__)
    for i in range(len(A__)):
        if A__[i] * B__[i] - OUT_[i] < 1:
            print("Mult:",A__[i] * B__[i],OUT_[i])
    print("Factored Number:",OUT_)
    print("Min. Hamilt.:", f)


end = time.time()
print(end - begin)

plt.plot(l)
# plt.legend()
plt.show()

plt.hist(f)
plt.show()