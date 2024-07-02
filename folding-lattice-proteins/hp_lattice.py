#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import defaultdict
import networkx as nx

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
    for i in evenpos:
        for u in g.nodes():
            if parity(u) == 0:
                QQ[((u, i), (u, i))] += -strength
        for u in g.nodes():
            for v in g.nodes():
                if u != v and parity(u) == 0 and parity(v) == 0:
                    QQ[((u, i), (v, i))] += strength
    for i in oddpos:
        for u in g.nodes():
            if parity(u) == 1:
                QQ[((u, i), (u, i))] += -strength
        for u in g.nodes():
            for v in g.nodes():
                if u != v and parity(u) == 1 and parity(v) == 1:
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
    def __init__(self, dim, sequence, Lambda=None):
        self.dim = dim
        if type(sequence) is str:
            self.sequence = from_str(sequence)
        else:
            self.sequence = sequence

        self.len_of_seq = len(sequence)

        if self.len_of_seq > np.prod(dim):
            raise RuntimeError(f"Lattice too small for sequence of length {self.len_of_seq}")

        # Lambda index 0: unique bead positions, 1: self avoidance 2: chain connectivity
        if Lambda is None:
            self.Lambda = [2.0, 3.0, 3.0]
            if self.len_of_seq >= 40:
                self.Lambda = [2.0, 3.5, 3.0] # known best values for S48
            if self.len_of_seq >= 60:
                self.Lambda = [3.0, 4.0, 4.0] # known best values for S64
        else:
            self.Lambda = Lambda

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
        print(f"Sequence (0 = P, 1 = H): {self.sequence}")
        print(f"Sequence length = {self.len_of_seq}")
        print(f"Lattice dimensions : {self.dim}")
        print("Created QBM with bit sequence keys.")
        print(f"bit vector has size {len(self.keys)}, each with {2*len(self.Q)/len(self.keys)} connections on average.")


    def interaction_matrix(self):
        return self.Q

    def print_energies(self, bits):
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
        print(f"EHP = {qhp}, E1 = {q1}, E2 = {q2}, E3 = {q3}, E = {qhp + q1 + q2 + q3}")

    def show_lattice(self, qubobitstring):
        latdim = self.dim
        image = np.zeros(latdim)
        for i in range(latdim[0]):
            for j in range (latdim[1]):
                image[i,j] = parity([i,j])
    
        colors = ["#eaeaea", "#fefefe"]
        lat_cmap = ListedColormap(colors, name="lat_cmap")
        hpcolors = ["#11f033", "#f03311"]
        hp_cmap = ListedColormap(hpcolors, name="hp_cmap")
        row_labels = range(latdim[0])
        col_labels = range(latdim[1])
        plt.matshow(image, cmap = lat_cmap)
        plt.xticks(range(latdim[1]), col_labels)
        plt.yticks(range(latdim[0]), row_labels)
    
        xpos = np.zeros(len(self.sequence))
        ypos = np.zeros(len(self.sequence))
        posc = np.zeros(len(self.sequence))
        xstart = []
        ystart = []
        cstart = []
        for i, b in enumerate(qubobitstring):
            if b == 0:
                continue
            s, f = self.keys[i]
            xpos[f] = (s[0])
            ypos[f] = (s[1])
            posc[f] = self.sequence[f]
            if f == 0:
                xstart.append(s[0])
                ystart.append(s[1])
                cstart.append(self.sequence[f])
        plt.scatter(xpos, ypos, s=100, c=posc, cmap=hp_cmap)
        plt.plot(xpos, ypos)
        plt.scatter(xstart, ystart, s=25, marker=5)
        plt.show()


