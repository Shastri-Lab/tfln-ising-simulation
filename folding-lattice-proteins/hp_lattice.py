"""
This file is copied and modified based on the file HP_Lattice_2D.py in Sandipan Mohanty's GitHub repo, found at:
https://github.com/sandipan-mohanty/DWaveHPLatticeProteins
"""
#%%

#!/usr/bin/env python
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from dimod.utilities import qubo_to_ising
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def parity(node):
    return (node[0] % 2 + node[1] % 2) % 2

def from_str(seqstr):
    ans = []
    for c in seqstr:
        if c == ' ':
            continue
        if c == 'H' or c == 'h' or c == '1':
            ans.append(1)
        else:
            ans.append(0)
    return ans

def to_str(seq_list):
    ans = ""
    for c in seq_list:
        if c == 1:
            ans += 'H'
        else:
            ans += 'P'
    return ans

def E_HP_qubo_contribs(g, sequence):
    QQ = defaultdict(float)
    for u, v in g.edges():
        if parity(u) == 0:
            ev, od = u, v
        else:
            ev, od = v, u
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                if (i - j) * (i - j) <= 4:
                    continue
                if (i + j) % 2 == 0: 
                    continue
                if (sequence[i] == 0 or sequence[j] == 0):
                    continue; 
                if (i % 2 == 0):
                    QQ[((ev, i), (od, j))] += -1
                else:
                    QQ[((ev, j), (od, i))] += -1

    return QQ

def constraint_unique_bead_location(g, strength, sequence_length):
    evenpos = [i for i in range(sequence_length) if i%2 == 0]
    oddpos = [i for i in range(sequence_length) if i%2 == 1]
    QQ = defaultdict(float)
    # for i in evenpos:
    #     for u in g.nodes():
    #         if parity(u) == 0:
    #             QQ[((u, i), (u, i))] += -strength
    #     # for u in g.nodes():
    #         for v in g.nodes():
    #             if u != v and parity(u) == 0 and parity(v) == 0:
    #                 QQ[((u, i), (v, i))] += strength
    # for i in oddpos:
    #     for u in g.nodes():
    #         if parity(u) == 1:
    #             QQ[((u, i), (u, i))] += -strength
    #     # for u in g.nodes():
    #         for v in g.nodes():
    #             if u != v and parity(u) == 1 and parity(v) == 1:
    #                 QQ[((u, i), (v, i))] += strength
    
    for i in range(sequence_length):
        for u in g.nodes():
            if parity(u) == i % 2:
                QQ[((u, i), (u, i))] += -strength
            for v in g.nodes():
                if u != v and parity(u) == i % 2 and parity(v) == i % 2:
                    QQ[((u, i), (v, i))] += strength

    return QQ

def constraint_self_avoidance(g, strength, sequence_length):
    evenpos = [i for i in range(sequence_length) if i%2 == 0]
    oddpos = [i for i in range(sequence_length) if i%2 == 1]
    QQ = defaultdict(float)
    for u in g.nodes():
        if parity(u) == 0:
            for x in evenpos:
                for y in evenpos:
                    if x < y:
                        QQ[((u, x), (u, y))] += strength
        else:
            for x in oddpos:
                for y in oddpos:
                    if x < y:
                        QQ[((u, x), (u, y))] += strength

    return QQ

def constraint_chain_connectivity(g, strength, sequence_length):
    evenpos = [i for i in range(sequence_length) if i%2 == 0]
    oddpos = [i for i in range(sequence_length) if i%2 == 1]
    QQ = defaultdict(float)
    for i in evenpos:
        for u in g.nodes():
            for v in g.nodes():
                if u == v or ((u,v) in g.edges()) or ((v,u) in g.edges()):
                    continue
                if parity(u) == 0 and parity(v) == 1:
                    if i != evenpos[-1] or sequence_length % 2 == 0:
                        QQ[((u, i), (v, i+1))] += strength
    for i in oddpos:
        for u in g.nodes():
            for v in g.nodes():
                if u == v or ((u,v) in g.edges()) or ((v,u) in g.edges()):
                    continue
                if parity(u) == 0 and parity(v) == 1:
                    if i != oddpos[-1] or sequence_length % 2 == 1:
                        QQ[((u, i+1), (v, i))] += strength
    return QQ

class Lattice_HP_QUBO:
    def __init__(self, dim, sequence, name=None, Lambda=None, target_energy=None, is_printing=True):
        self.name = "X" if name is None else name
        self.dim = dim
        self.target_energy = target_energy
        
        if type(sequence) is str:
            self.sequence = from_str(sequence)
        else:
            self.sequence = sequence
        self.len_of_seq = len(sequence)
        if self.len_of_seq > np.prod(dim):
            raise RuntimeError(f"Lattice too small for sequence of length {self.len_of_seq}")

        if Lambda is None: # Lambda index 0: unique bead positions, 1: self avoidance 2: chain connectivity
            self.Lambda = [2.0, 3.0, 3.0]
            if self.len_of_seq >= 40:
                self.Lambda = [2.0, 3.5, 3.0] # known best values for S48
            if self.len_of_seq >= 60:
                self.Lambda = [3.0, 4.0, 4.0] # known best values for S64
        else:
            self.Lambda = Lambda

        print(f'in __init__: {self.Lambda=}')

        G = nx.grid_graph(self.dim)
        self.Q = defaultdict(float)

        self.QHP = E_HP_qubo_contribs(G, self.sequence)
        self.Q1 = constraint_unique_bead_location(G, self.Lambda[0], self.len_of_seq)
        self.Q2 = constraint_self_avoidance(G, self.Lambda[1], self.len_of_seq)
        self.Q3 = constraint_chain_connectivity(G, self.Lambda[2], self.len_of_seq)
        for vpair in self.QHP:
            self.Q[vpair] += self.QHP[vpair]
        for vpair in self.Q1:
            self.Q[vpair] += self.Q1[vpair]
        for vpair in self.Q2:
            self.Q[vpair] += self.Q2[vpair]
        for vpair in self.Q3:
            self.Q[vpair] += self.Q3[vpair]
        ukeys = []
        for k in self.Q:
            ukeys.append(k[0])
            ukeys.append(k[1])
        self.keys = list(sorted(set(ukeys)))
        if is_printing:
            print(f"Sequence: {self.seq_to_str()}")
            print(f"Sequence length = {self.len_of_seq}")
            print(f"Lattice dimensions : {self.dim}")
            print(f"Bit vector has size {len(self.keys)}, each with {2*len(self.Q)/len(self.keys):.2f} connections on average.")
            print(f'Q contains elements in {set(self.Q.values())}')

    def seq_to_str(self):
        return to_str(self.sequence)

    def interaction_matrix(self):
        return self.Q

    def optimization_matrix(self):
        return self.QHP

    def constraint_matrix_1(self):
        return self.Q1
    
    def constraint_matrix_2(self):
        return self.Q2
    
    def constraint_matrix_3(self):
        return self.Q3

    def Q_as_np_array(self, Q_dict):
        Q = np.zeros((len(self.keys), len(self.keys)))
        for i in range(len(self.keys)):
            for j in range(len(self.keys)):
                if (self.keys[i], self.keys[j]) in Q_dict:
                    Q[i, j] = Q_dict[(self.keys[i], self.keys[j])]
        return Q

    def get_energies(self, bits):
        # print(f'in get_energies: {self.Lambda=}')
        qhp = 0.
        q1 = self.Lambda[0] * self.len_of_seq
        q2 = 0. 
        q3 = 0.

        for i in range(len(bits)):
            if bits[i] == 0:
                continue
            for j in range(len(bits)):
                if bits[j] == 0:
                    continue
                qhp += self.QHP[(self.keys[i], self.keys[j])]
                q1 += self.Q1[(self.keys[i], self.keys[j])]
                q2 += self.Q2[(self.keys[i], self.keys[j])]
                q3 += self.Q3[(self.keys[i], self.keys[j])]
        return qhp, q1, q2, q3

    def print_energies(self, bits):
        qhp, q1, q2, q3 = self.get_energies(bits)
        print(f"EHP = {qhp}, E1 = {q1}, E2 = {q2}, E3 = {q3}, E = {qhp + q1 + q2 + q3}")

    def to_ising(self):
        h_dict, J_dict, offset_ising = qubo_to_ising(self.interaction_matrix())
        return h_dict, J_dict, offset_ising

    def show_lattice(self, qubobitstring, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(8, 8))

        # plot checkerboard background for lattice
        latdim = self.dim
        image = np.zeros(latdim)
        for i in range(latdim[0]):
            for j in range (latdim[1]):
                image[i,j] = parity([i,j])
        colors = ["#eaeaea", "#fefefe"]
        lat_cmap = ListedColormap(colors, name="lat_cmap")
        row_labels = range(latdim[0])
        col_labels = range(latdim[1])
        axes.matshow(image, cmap = lat_cmap)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xticklabels([])
        axes.set_yticklabels([])
    
        # plot the amino acids on the lattice
        hpcolors = ["#11f033", "#f03311"]
        hp_cmap = ListedColormap(hpcolors, name="hp_cmap")
        fpos = []
        xpos = [] #np.zeros(len(self.sequence))
        ypos = [] #np.zeros(len(self.sequence))
        posc = [] #np.zeros(len(self.sequence))
        xstart = []
        ystart = []
        cstart = []
        text_dict = {}
        for i, b in enumerate(qubobitstring):
            if b != 1:
                continue
            s, f = self.keys[i]
            fpos.append(f)
            xpos.append(s[0])
            ypos.append(s[1])
            posc.append(self.sequence[f])
            if f == 0:
                xstart.append(s[0])
                ystart.append(s[1])
                cstart.append(self.sequence[f])

            t = text_dict.get((s[0], s[1]), [])
            t.append(f)
            text_dict[(s[0], s[1])] = t            
            
        for k, v in text_dict.items():
            t = ""
            for i in v:
                t += str(i) + ","
            t = t[:-1]
            axes.text(k[0]-0.4, k[1]-0.3, t, color='k', fontsize=8, ha='left')

        # sort xpos and ypos based on fpos
        xpos = np.array([x for _, x in sorted(zip(fpos, xpos), key=lambda a: a[0])])
        ypos = np.array([y for _, y in sorted(zip(fpos, ypos), key=lambda a: a[0])])
        posc = np.array([p for _, p in sorted(zip(fpos, posc), key=lambda a: a[0])])

        axes.plot(xpos, ypos, 'k-', lw=0.5)
        axes.scatter(xpos, ypos, s=100, c=posc, cmap=hp_cmap, alpha=0.5)
        axes.scatter(xstart, ystart, c='k', s=25, marker=5)


# %%
