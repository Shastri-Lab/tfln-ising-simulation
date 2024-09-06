import numpy as np


# Constructs the Hamiltonian to factorize F = I1*I2

def construct_J(m):
    # m = len(np.binary_repr(np.maximum(I1,I2)))
    n = 2*m
    N = 3*m**2

    J_AB = -1 * np.ones((m, m))

    J_ABS = np.zeros(n)
    J_ABS[[0, m]] = 2

    J_SS = np.zeros((n, n))
    J_SS[-1, -2] = -2

    J_AV1 = np.zeros((m, 2 * m - 1))
    J_AVn = np.zeros((m, 2 * m - 1))
    for i in range(m):
        if i == 0:
            J_AV1[i, i] = 2
            J_AVn[i, 2 * i] = 2
        else:
            J_AV1[i, [2 * i - 1, 2 * i]] = 2
            J_AVn[i, 2 * i] = 2

    J_BV1 = np.zeros((2, m ** 2 - 1))
    J_BV1[0, np.arange(1, 2 * m - 1, 2)] = 2
    J_BV1[1, np.arange(0, 2 * m, 2)] = 2

    J_BVn = 2 * np.zeros(2 * m - 1)
    J_BVn[np.arange(0, 2 * m - 1, 2)] = 2

    J = np.zeros((N, N))

    J[m:n, :m] = J_AB
    J[n, :n] = J_ABS
    J[n:2 * n, n:2 * n] = J_SS
    J[:m, 2 * n:2 * n + 2 * m - 1] = J_AV1
    J[m:m + 2, 2 * n:2 * n + m ** 2 - 1] = J_BV1
    if m > 2:
        for i in range(1, m - 1):
            J[:m, 2 * n + 2 * m - 1 + (i - 1) * (2 * m - 1):2 * n + 2 * m - 1 + i * (2 * m - 1)] = J_AVn
            J[m + 1 + i, 2 * n + 2 * m - 1 + (i - 1) * (2 * m - 1):2 * n + 2 * m - 1 + i * (2 * m - 1)] = J_BVn

    # J_SV
    J[2 * m + 1, [2 * n, 2 * n + 1]] = 1  # SV1
    if m > 2:
        for i in range(m - 3):  # SVn, n != -1
            J[2 * m + 1 + i + 1, 2 * n + 2 * m - 1 + i * (2 * m - 1)] = 1
            J[2 * m + 1 + i + 1, 2 * n + 2 * m - 1 + i * (2 * m - 1) + 1] = 1
    for i in range(m - 1, n - 1):
        J[n + i, 2 * n + (2 * m - 1) * (m - 2) + 2 * (i - m + 1)] = 1  # SVn
        if i != n - 2:
            J[n + i, 2 * n + (2 * m - 1) * (m - 2) + 2 * (i - m + 1) + 1] = 1
        if i == n - 2:
            J[n + i + 1, 2 * n + (2 * m - 1) * (m - 2) + 2 * (i - m + 1)] = 2

    # J_SH
    for i in range(1, m):
        J[n + i, 2 * n + (n - 1) * (m - 1) + (m - 1) * (i - 1)] = -2  # SH1..SH1m-2
    for i in range(m - 2):
        J[n + m + i, 2 * n + (n - 1) * (m - 1) + (m - 1) * (m - 2) + i] = 1
        J[n + m + i, 2 * n + (n - 1) * (m - 1) + (m - 1) * (m - 2) + i + 1] = -2
    J[2 * n - 2, 2 * n + (n - 1) * (m - 1) + (m - 1) ** 2 - 1] = 1
    J[2 * n - 1, 2 * n + (n - 1) * (m - 1) + (m - 1) ** 2 - 1] = 2

    # J_SC
    if m > 2:
        J[2 * n - 1, -1] = 2
        J[2 * n - 2, -1] = 1

    # J_VV
    J_VV = np.array([[-1 if (i == j - 1 and j % 2 == 1) else 0 for i in range(2 * m - 1)] for j in range(2 * m - 1)])
    for i in range(m - 1):
        J[2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1),
        2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1)] = J_VV
    J_VV = np.array([[1 if (i - 1 == j and i % 2 == 0) else 0 for i in range(2 * m - 1)] for j in range(2 * m - 1)])
    J_VV += np.array([[1 if (i - 2 == j and i % 2 == 1) else 0 for i in range(2 * m - 1)] for j in range(2 * m - 1)])
    for i in range(1, m - 1):
        J[2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1),
        2 * n + (i - 1) * (2 * m - 1):2 * n + i * (2 * m - 1)] = J_VV

    # J_VH
    J_VH = np.zeros((m - 1, 2 * m - 1))
    for i in range(m - 2):
        J_VH[i, 2 * i:2 * i + 4] = np.array([2, 2, -1, -1])
    J_VH[-1, -3:] = np.array([2, 2, -1])
    for i in range(m - 1):
        J[2 * n + (2 * m - 1) * (m - 1) + i * (m - 1):2 * n + (2 * m - 1) * (m - 1) + (i + 1) * (m - 1),
        2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1)] = J_VH
    J_VH = np.array([[1 if (int(i / 2) == j and i % 2 == 1) else 0 for i in range(2 * m - 1)] for j in range(m - 1)])
    J_VH += np.array(
        [[-2 if (int(i / 2) + 1 == j and i % 2 == 1) else 0 for i in range(2 * m - 1)] for j in range(m - 1)])
    for i in range(1, m - 1):
        J[2 * n + (2 * m - 1) * (m - 1) + (i - 1) * (m - 1):2 * n + (2 * m - 1) * (m - 1) + i * (m - 1),
        2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1)] = J_VH

    # J_CV
    if m > 2:
        for i in range(m - 3):
            inds = np.array([(i + 1) * (2 * m - 1) - 1, (i + 2) * (2 * m - 1) - 2, (i + 2) * (2 * m - 1) - 1,
                             (i + 3) * (2 * m - 1) - 2])
            J[2 * n + (m - 1) * (2 * m - 1) + (m - 1) ** 2 + i, 2 * n + inds] = [2, -2, -1, 1]
        inds = np.array([(m - 2) * (2 * m - 1) - 1, (m - 1) * (2 * m - 1) - 2, (m - 1) * (2 * m - 1) - 1])
        J[-1, 2 * n + inds] = [2, -2, -1]

    # J_HH
    J_HH = np.array([[2 if (i == j - 1) else 0 for i in range(m - 1)] for j in range(m - 1)])
    for i in range(m - 1):
        J[2 * n + (2 * m - 1) * (m - 1) + i * (m - 1):2 * n + (2 * m - 1) * (m - 1) + (i + 1) * (m - 1),
        2 * n + (2 * m - 1) * (m - 1) + i * (m - 1):2 * n + (2 * m - 1) * (m - 1) + (i + 1) * (m - 1)] = J_HH

    # J_HCs
    if m > 2:
        for i in range(m - 2):
            J[2 * n + (2 * m - 1) * (m - 1) + (m - 1) ** 2 + i, 2 * n + (2 * m - 1) * (m - 1) + (i + 1) * (
                        m - 1) - 1] = 2
            J[2 * n + (2 * m - 1) * (m - 1) + (m - 1) ** 2 + i, 2 * n + (2 * m - 1) * (m - 1) + (i + 2) * (
                        m - 1) - 1] = -1

    # J_CC
    if m > 3:
        J_CC = np.array([[2 if (i == j - 1) else 0 for i in range(m - 2)] for j in range(m - 2)])
        J[N - (m - 2):, N - (m - 2):] = J_CC

    return J + J.T


def construct_h(m):
    n = 2*m
    N = 3*m**2
    h = np.zeros(N)
    # A
    # print(0,m)
    h[:m] += m
    # B
    # print(m,2*m)
    h[m:2 * m] += m
    # S
    # print(n)
    h[n] += -2
    # print(n+1,n+1+m-1)
    h[n + 1:n + 1 + m - 1] += -1
    if m == 2:
        h[n + n - 2] += -1
        h[n + n - 1] += -2
    # V
    # Contributions from above
    # print(2*n,2*n+(2*m-1))
    h[2 * n:2 * n + (2 * m - 1)] += -2  # line 1
    for i in range(1, m - 1):  # line 2...m-1
        # print(2*n+i*(2*m-1),2*n+(i+1)*(2*m-1),np.arange(0,2*m-1,2))
        h[2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1)][np.arange(0, 2 * m - 1, 2)] += -2  # AND
        if i == 1:
            # print(2*n+i*(2*m-1),2*n+(i+1)*(2*m-1),-2)
            h[2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1)][[-2]] += -1  # HA S
    # Contributions from below
    # print(2*n,2*n+(2*m-1),[0,1,-1])
    h[2 * n:2 * n + (2 * m - 1)][[0, 1, -1]] += 1  # HA A,B
    for i in range(1, m - 1):  # line 2...m-1
        # print(2*n+i*(2*m-1),2*n+(i+1)*(2*m-1),[0,1])
        h[2 * n + i * (2 * m - 1):2 * n + (i + 1) * (2 * m - 1)][[0, 1]] += 1  # HA A,B
    # H
    for i in range(m - 1):
        h[2 * n + (2 * m - 1) * (m - 1) + i * (m - 1):2 * n + (2 * m - 1) * (m - 1) + (i + 1) * (m - 1)][
            0] += -2  # HA CO
        if i == 0:
            h[2 * n + (2 * m - 1) * (m - 1) + i * (m - 1):2 * n + (2 * m - 1) * (m - 1) + (i + 1) * (m - 1)][
                -1] += 1  # HA B
    # C
    if m > 2:
        h[2 * n + (2 * m - 1) * (m - 1) + (m - 1) ** 2] += -2  # HA CO

    return h