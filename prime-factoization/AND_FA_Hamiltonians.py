import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time

def AND(m1,m2,m3,Jarr,hvec):
    Jarr[m1,m2] += -1
    Jarr[m2,m1] += -1
    Jarr[m1,m3] += 2
    Jarr[m3,m1] += 2
    Jarr[m2,m3] += 2
    Jarr[m3,m2] += 2
    hvec[m1,0] += 1
    hvec[m2,0] += 1
    hvec[m3,0] += -2
    return Jarr, hvec

def FA(A,B,Cin,S,C,Jarr,hvec):
    m1 = A
    m2 = B
    m3 = Cin
    m4 = S
    m5 = C
    Jarr[m1, m2] += -1
    Jarr[m2, m1] += -1
    Jarr[m1, m3] += -1
    Jarr[m3, m1] += -1
    Jarr[m1, m4] += 1
    Jarr[m4, m1] += 1
    Jarr[m1, m5] += 2
    Jarr[m5, m1] += 2
    Jarr[m2, m3] += -1
    Jarr[m3, m2] += -1
    Jarr[m2, m4] += 1
    Jarr[m4, m2] += 1
    Jarr[m2, m5] += 2
    Jarr[m5, m2] += 2
    Jarr[m3, m4] += 1
    Jarr[m4, m3] += 1
    Jarr[m3, m5] += 2
    Jarr[m5, m3] += 2
    Jarr[m4, m5] += -2
    Jarr[m5, m4] += -2
    return Jarr, hvec

def generate_Nbit_sparse_multiplier_Hamiltonian(Nbits_in):
    N_spins = 3 * Nbits_in ** 2 + Nbits_in

    Out_count = 0 + 2 * Nbits_in
    J_glob = sp.lil_array((N_spins,N_spins))
    h_glob = sp.lil_array((N_spins,1))
    J_glob,h_glob = AND(0, Nbits_in, Out_count, J_glob,h_glob)
    Out_count += 1
    dummy = 0 # auxilliary bits counter
    dummy_start = 4 * Nbits_in # auxilliary bits starting spot in spin vector

    # First column of AND gates
    for i in range(Nbits_in):
        for j in range(2):
            j_ = j + Nbits_in
            if not (i == 0 and j == 0):
                J_glob,h_glob = AND(i, j_, dummy + dummy_start, J_glob,h_glob)
                dummy += 1
    # print(dummy)
    dummy += 1  # include null input at bottom of first column

    # Remaining AND gates
    for j in range(2, Nbits_in):
        j_ = j + Nbits_in
        for i in range(Nbits_in):
            J_, h_ = AND(i, j_, dummy + dummy_start, J_glob,h_glob)
            dummy += 1

    # print(dummy)

    # first row of FAs
    for i in range(Nbits_in):
        if i == 0:
            J_glob,h_glob = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy, Out_count,
                            dummy_start + dummy + 1, J_glob,h_glob)  # outputs P1
            Out_count += 1
        else:
            if i == Nbits_in - 1:
                J_glob,h_glob = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + Nbits_in, J_glob,h_glob)
            else:
                J_glob,h_glob = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + 1, J_glob,h_glob)
        dummy += 1

    dummy += Nbits_in  # 4 outputs carried to next row

    # print(dummy)
    # print('out count',Out_count - 2*Nbits_in)

    # FAs in Middle rows
    for j in range(Nbits_in - 3):
        for i in range(Nbits_in):
            if i == 0:
                J_glob,h_glob = FA(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                Out_count, dummy_start + dummy + 1, J_glob,h_glob)
                Out_count += 1
            else:
                if i == Nbits_in - 1:
                    J_glob,h_glob = FA(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                    dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + Nbits_in, J_glob,h_glob)
                else:
                    J_glob,h_glob = FA(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                    dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + 1, J_glob,h_glob)

            dummy += 1

        dummy += Nbits_in  # 4 outputs carried to next row

    # print(dummy)
    # print('out count',Out_count - 2*Nbits_in)

    for i in range(Nbits_in):  # last row
        if i == Nbits_in - 1:
            J_glob,h_glob = FA(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, Out_count + 1, J_glob,h_glob)
        else:
            J_glob,h_glob = FA(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, dummy_start + dummy + 1, J_glob,h_glob)
            Out_count += 1
        dummy += 1

    # print('out count',Out_count - 2*Nbits_in)
    return J_glob, h_glob


def generate_Nbit_multiplier_Hamiltonian(Nbits_in):
    N_spins = 3 * Nbits_in ** 2 + Nbits_in

    Out_count = 0 + 2 * Nbits_in
    J_glob = np.zeros((N_spins,N_spins))
    h_glob = np.zeros((N_spins,1))

    J_glob,h_glob = AND(0, Nbits_in, Out_count, J_glob,h_glob)
    Out_count += 1

    dummy = 0 # auxilliary bits counter
    dummy_start = 4 * Nbits_in # auxilliary bits starting spot in spin vector

    # First column of AND gates
    for i in range(Nbits_in):
        for j in range(2):
            j_ = j + Nbits_in
            if not (i == 0 and j == 0):
                J_glob,h_glob = AND(i, j_, dummy + dummy_start, J_glob,h_glob)
                dummy += 1
    # print(dummy)
    dummy += 1  # include null input at bottom of first column

    # Remaining AND gates
    for j in range(2, Nbits_in):
        j_ = j + Nbits_in
        for i in range(Nbits_in):
            J_, h_ = AND(i, j_, dummy + dummy_start, J_glob,h_glob)
            dummy += 1

    # print(dummy)

    # first row of FAs
    for i in range(Nbits_in):
        if i == 0:
            J_glob,h_glob = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy, Out_count,
                            dummy_start + dummy + 1, J_glob,h_glob)  # outputs P1
            Out_count += 1
        else:
            if i == Nbits_in - 1:
                J_glob,h_glob = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + Nbits_in, J_glob,h_glob)
            else:
                J_glob,h_glob = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + 1, J_glob,h_glob)
        dummy += 1

    dummy += Nbits_in  # 4 outputs carried to next row

    # print(dummy)
    # print('out count',Out_count - 2*Nbits_in)

    # FAs in Middle rows
    for j in range(Nbits_in - 3):
        for i in range(Nbits_in):
            if i == 0:
                J_glob,h_glob = FA(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                Out_count, dummy_start + dummy + 1, J_glob,h_glob)
                Out_count += 1
            else:
                if i == Nbits_in - 1:
                    J_glob,h_glob = FA(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                    dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + Nbits_in, J_glob,h_glob)
                else:
                    J_glob,h_glob = FA(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                    dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + 1, J_glob,h_glob)

            dummy += 1

        dummy += Nbits_in  # 4 outputs carried to next row

    # print(dummy)
    # print('out count',Out_count - 2*Nbits_in)

    for i in range(Nbits_in):  # last row
        if i == Nbits_in - 1:
            J_glob,h_glob = FA(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, Out_count + 1, J_glob,h_glob)
        else:
            J_glob,h_glob = FA(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, dummy_start + dummy + 1, J_glob,h_glob)
            Out_count += 1
        dummy += 1

    # print('out count',Out_count - 2*Nbits_in)
    return J_glob, h_glob

begin = time.time()
J, h = generate_Nbit_sparse_multiplier_Hamiltonian(215)
end = time.time()

print(end - begin)
print(J.shape,h.shape)
print(J.nnz)
print(J.nnz / J.shape[0])
# print(np.max(np.max(J)))
# print(np.min(np.min(J)))
J = J.tocsr()
h = h.tocsr()

print(J.min())
print(J.max())
print(J[J.nonzero()].shape[0] / J.shape[0])
Jnz = J[J.nonzero()]
hnz = h[h.nonzero()]

plt.hist(Jnz, bins = np.arange(np.min(Jnz) - 0.5, np.max(Jnz) + 1.5))
plt.xlabel("values of J matrix")
plt.ylabel("count")
plt.tight_layout()
plt.show()
plt.hist(hnz, bins = np.arange(np.min(hnz)- 0.5, np.max(hnz) + 1.5))
plt.xlabel("values of h vector")
plt.ylabel("count")
plt.tight_layout()
plt.show()

def generate_Nbit_multiplier_Hamiltonian_old(Nbits_in):
    N_spins = 3 * Nbits_in ** 2 + Nbits_in

    Out_count = 0 + 2 * Nbits_in

    J1, h1 = AND_old(0, Nbits_in, Out_count, N_spins)
    Out_count += 1

    J_glob = J1.copy()
    h_glob = h1.copy()
    dummy = 0 # auxilliary bits counter
    dummy_start = 4 * Nbits_in # auxilliary bits starting spot in spin vector

    # First column of AND gates
    for i in range(Nbits_in):
        for j in range(2):
            j_ = j + Nbits_in
            if not (i == 0 and j == 0):
                J_, h_ = AND_old(i, j_, dummy + dummy_start, N_spins)
                dummy += 1
                J_glob += J_
                h_glob += h_
    # print(dummy)
    dummy += 1  # include null input at bottom of first column

    # Remaining AND gates
    for j in range(2, Nbits_in):
        j_ = j + Nbits_in
        for i in range(Nbits_in):
            J_, h_ = AND_old(i, j_, dummy + dummy_start, N_spins)
            dummy += 1
            J_glob += J_
            h_glob += h_

    last_AND_dummy = dummy
    # print(dummy)

    # first row of FAs
    for i in range(Nbits_in):
        if i == 0:
            J_FA, h_FA = FA_old(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy, Out_count,
                            dummy_start + dummy + 1, N_spins)  # outputs P1
            Out_count += 1
        else:
            if i == Nbits_in - 1:
                J_FA, h_FA = FA_old(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + Nbits_in, N_spins)
            else:
                J_FA, h_FA = FA_old(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + 1, N_spins)
        dummy += 1
        J_glob += J_FA
        h_glob += h_FA

    dummy += Nbits_in  # 4 outputs carried to next row

    # print(dummy)
    # print('out count',Out_count - 2*Nbits_in)

    # FAs in Middle rows
    for j in range(Nbits_in - 3):
        for i in range(Nbits_in):
            if i == 0:
                J_FA, h_FA = FA_old(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                Out_count, dummy_start + dummy + 1, N_spins)
                Out_count += 1
            else:
                if i == Nbits_in - 1:
                    J_FA, h_FA = FA_old(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                    dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + Nbits_in, N_spins)
                else:
                    J_FA, h_FA = FA_old(dummy_start + (2 + j) * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                    dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + 1, N_spins)

            dummy += 1
            J_glob += J_FA
            h_glob += h_FA

        dummy += Nbits_in  # 4 outputs carried to next row

    # print(dummy)
    # print('out count',Out_count - 2*Nbits_in)

    for i in range(Nbits_in):  # last row
        if i == Nbits_in - 1:
            J_FA, h_FA = FA_old(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, Out_count + 1, N_spins)
        else:
            J_FA, h_FA = FA_old(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, dummy_start + dummy + 1, N_spins)
            Out_count += 1
        dummy += 1

        J_glob += J_FA
        h_glob += h_FA
    # print('out count',Out_count - 2*Nbits_in)
    return J_glob, h_glob


def generate_4bit_multiplier_Hamiltonian(Nbits_in):
    N_spins = 3 * Nbits_in ** 2 + Nbits_in

    Out_count = 0 + 2 * Nbits_in

    J1, h1 = AND(0, Nbits_in, Out_count, N_spins)
    Out_count += 1

    J_glob = J1.copy()
    h_glob = h1.copy()
    dummy = 0
    dummy_start = 4 * Nbits_in

    for i in range(Nbits_in):
        for j in range(2):
            j_ = j + Nbits_in
            if not (i == 0 and j == 0):
                J_, h_ = AND(i, j_, dummy + dummy_start, N_spins)
                dummy += 1
                J_glob += J_
                h_glob += h_
    print(dummy)
    dummy += 1  # include null input at bottom of first column

    for j in range(2, Nbits_in):
        j_ = j + Nbits_in
        for i in range(Nbits_in):
            J_, h_ = AND(i, j_, dummy + dummy_start, N_spins)
            dummy += 1
            J_glob += J_
            h_glob += h_

    last_AND_dummy = dummy
    print(dummy)

    for i in range(Nbits_in):  # first row
        if i == 0:
            J_FA, h_FA = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy, Out_count,
                            dummy_start + dummy + 1, N_spins)  # outputs P1
            Out_count += 1
        else:
            if i == Nbits_in - 1:
                J_FA, h_FA = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + Nbits_in, N_spins)
            else:
                J_FA, h_FA = FA(dummy_start + 2 * i, dummy_start + 2 * i + 1, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1),
                                dummy_start + dummy + 1, N_spins)

        dummy += 1
        J_glob += J_FA
        h_glob += h_FA

    dummy += Nbits_in  # 4 outputs carried to next row

    print(dummy)

    for i in range(Nbits_in):  # second row
        if i == 0:
            J_FA, h_FA = FA(dummy_start + 2 * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                            Out_count, dummy_start + dummy + 1, N_spins)
            Out_count += 1
        else:
            if i == Nbits_in - 1:
                J_FA, h_FA = FA(dummy_start + 2 * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + Nbits_in, N_spins)
            else:
                J_FA, h_FA = FA(dummy_start + 2 * Nbits_in + i, dummy_start + dummy - Nbits_in, dummy_start + dummy,
                                dummy_start + dummy + (Nbits_in - 1), dummy_start + dummy + 1, N_spins)

        dummy += 1
        J_glob += J_FA
        h_glob += h_FA

    dummy += Nbits_in  # 4 outputs carried to next row

    print(dummy)

    for i in range(Nbits_in):  # last row
        if i == Nbits_in - 1:
            J_FA, h_FA = FA(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, Out_count + 1, N_spins)
        else:
            J_FA, h_FA = FA(dummy_start + (Nbits_in ** 2 - Nbits_in) + i, dummy_start + dummy - Nbits_in,
                            dummy_start + dummy,
                            Out_count, dummy_start + dummy + 1, N_spins)
            Out_count += 1
        dummy += 1

        J_glob += J_FA
        h_glob += h_FA
    return J_glob, h_glob

# def AND(m1,m2,m3,Jarr,hvec):
#     Jarr[m1,m2] += -1
#     Jarr[m2,m1] += -1
#     Jarr[m1,m3] += 2
#     Jarr[m3,m1] += 2
#     Jarr[m2,m3] += 2
#     Jarr[m3,m2] += 2
#     hvec[m1] += 1
#     hvec[m2] += 1
#     hvec[m3] += -2
#     return Jarr, hvec
#
# def FA(A,B,Cin,S,C,Jarr,hvec):
#     m1 = A
#     m2 = B
#     m3 = Cin
#     m4 = S
#     m5 = C
#     Jarr[m1, m2] += -1
#     Jarr[m2, m1] += -1
#     Jarr[m1, m3] += -1
#     Jarr[m3, m1] += -1
#     Jarr[m1, m4] += 1
#     Jarr[m4, m1] += 1
#     Jarr[m1, m5] += 2
#     Jarr[m5, m1] += 2
#     Jarr[m2, m3] += -1
#     Jarr[m3, m2] += -1
#     Jarr[m2, m4] += 1
#     Jarr[m4, m2] += 1
#     Jarr[m2, m5] += 2
#     Jarr[m5, m2] += 2
#     Jarr[m3, m4] += 1
#     Jarr[m4, m3] += 1
#     Jarr[m3, m5] += 2
#     Jarr[m5, m3] += 2
#     Jarr[m4, m5] += -2
#     Jarr[m5, m4] += -2
#     return Jarr, hvec

def AND_old(m1,m2,m3,Nspins):
    Jarr = np.zeros((Nspins,Nspins))
    hvec = np.zeros(Nspins)
    Jarr[m1,m2] += -1
    Jarr[m2,m1] += -1
    Jarr[m1,m3] += 2
    Jarr[m3,m1] += 2
    Jarr[m2,m3] += 2
    Jarr[m3,m2] += 2
    hvec[m1] += 1
    hvec[m2] += 1
    hvec[m3] += -2
    return Jarr, hvec

def FA_old(A,B,Cin,S,C,Nspins):
    Jarr = np.zeros((Nspins,Nspins))
    hvec = np.zeros(Nspins)
    m1 = A
    m2 = B
    m3 = Cin
    m4 = S
    m5 = C
    Jarr[m1, m2] += -1
    Jarr[m2, m1] += -1
    Jarr[m1, m3] += -1
    Jarr[m3, m1] += -1
    Jarr[m1, m4] += 1
    Jarr[m4, m1] += 1
    Jarr[m1, m5] += 2
    Jarr[m5, m1] += 2
    Jarr[m2, m3] += -1
    Jarr[m3, m2] += -1
    Jarr[m2, m4] += 1
    Jarr[m4, m2] += 1
    Jarr[m2, m5] += 2
    Jarr[m5, m2] += 2
    Jarr[m3, m4] += 1
    Jarr[m4, m3] += 1
    Jarr[m3, m5] += 2
    Jarr[m5, m3] += 2
    Jarr[m4, m5] += -2
    Jarr[m5, m4] += -2
    return Jarr, hvec
# print(AND(0,1,2,3))
# print(FA(0,1,2,3,4,5))


# J_And = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
# h_and = np.array([1,1,-2])
# i1 = [-1,1,-1,1,-1,1,-1,1]
# i2 = [-1,-1,1,1,-1,-1,1,1]
# o = [-1,-1,-1,-1,1,1,1,1]
#
# Hamiltonian = []
# spin_vecs = []
# for i,j,k in zip(i1,i2,o):
#     spin_vec = np.array([i,j,k])
#     spin_vecs.append(str(i + 1) + str(j + 1) + str(k + 1))
#     energy = 0.5 * np.sum(spin_vec * (J_And @ spin_vec.transpose())) + np.sum(h_and * spin_vec)
#     Hamiltonian.append(energy)
#     print(J_And @ spin_vec)
#
# plt.bar(spin_vecs,Hamiltonian)
# plt.show()

#
# J_And, h_and = FA(0,1,2,3,4,5)
# co = [-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1]
# su = [-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1]
# ci = [-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1]
# i2 = [-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]
# i1 = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#
# Hamiltonian = []
# spin_vecs = []
# for i,j,k,l,m in zip(i1,i2,ci,su,co):
#     spin_vec = np.array([i,j,k,l,m])
#     spin_vecs.append(str(i + 1) + str(j + 1) + str(k + 1) + str(l + 1) + str(m + 1))
#     energy = 0.5 * np.sum(spin_vec * (J_And @ spin_vec.transpose())) + np.sum(h_and * spin_vec)
#     Hamiltonian.append(energy)
#     print(J_And @ spin_vec)
#
# plt.figure(figsize=[20,8])
# plt.bar(spin_vecs,Hamiltonian)
# plt.xticks(rotation=30)
# plt.show()
