import numpy as np

def generate_even_sum_set(size, max_value):
    nset = np.random.randint(1, max_value, size)
    if np.sum(nset) % 2 != 0:
        nset[-1] += 1
        if nset[-1] > max_value:
            nset[-1] -= 2
    return nset.astype(np.float64)

def ising_number_partitioning_matrix(nset):
    N = len(nset)
    nset /= nset.max()
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            J[i, j] = nset[i] * nset[j]
            J[j, i] = J[i, j]
    return J
