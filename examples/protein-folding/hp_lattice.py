"""
This file is derived from the script HP_Lattice_2D.py originally developed by Sandipan Mohanty and collaborators for "Folding lattice proteins with quantum annealing" (Phys. Rev. Research, vol. 4, no. 4, p. 043013, Oct. 2022, doi: 10.1103/PhysRevResearch.4.043013), available at https://github.com/sandipan-mohanty/DWaveHPLatticeProteins

The present version was independently refactored and extended by Hugh Morison and collaborators for related academic work. This code is not affiliated with or endorsed by the original authors. All original rights remain with them.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dimod.utilities import qubo_to_ising
from matplotlib.colors import ListedColormap

class HP_Lattice_Problem:
    """Class representing the QUBO formulation for HP lattice protein folding."""

    def __init__(self, dim, sequence, name=None, lambd=None, target_energy=None, is_printing=True):
        self.name = name if name is not None else "unnamed"
        self.lattice_dimensions = dim
        self.target_energy = target_energy

        self.sequence = self.sequence_from_string(sequence) if isinstance(sequence, str) else sequence
        self.sequence_length = len(self.sequence)
        if self.sequence_length > np.prod(dim):
            raise RuntimeError(f"Lattice too small for sequence of length {self.sequence_length}")

        if lambd is None: # parameters are best from "Folding Lattice Proteins w/ Quantum Annealing" paper
            if self.sequence_length >= 60:
                self.lambd = [3.0, 4.0, 4.0]  # S64
            elif self.sequence_length >= 40:
                self.lambd = [2.0, 3.5, 3.0]  # S48
            else:
                self.lambd = [2.0, 3.0, 3.0]
        else:
            self.lambd = lambd

        square_lattice_graph = nx.grid_graph(self.lattice_dimensions)

        # QUBO matrices
        self.Q = defaultdict(float)
        self.QHP = self.get_dict_HP_energy(square_lattice_graph, self.sequence)
        self.Q1 = self.get_dict_unique_location(square_lattice_graph, self.lambd[0], self.sequence_length)
        self.Q2 = self.get_dict_self_avoidance(square_lattice_graph, self.lambd[1], self.sequence_length)
        self.Q3 = self.get_dict_chain_connectivity(square_lattice_graph, self.lambd[2], self.sequence_length)

        # Aggregate all QUBO terms
        for Q_dict in [self.QHP, self.Q1, self.Q2, self.Q3]:
            for key, value in Q_dict.items():
                self.Q[key] += value

        # Collect all unique keys
        ukeys = [k for pair in self.Q for k in pair]
        self.keys = sorted(set(ukeys))

        if is_printing:
            print(f"Sequence: {self.seq_to_str()}")
            print(f"Sequence length: {self.sequence_length}")
            print(f"Lattice dimensions: {self.lattice_dimensions}")
            avg_conn = 2 * len(self.Q) / len(self.keys) if self.keys else 0
            print(f"Bit vector size: {len(self.keys)}, avg. connections: {avg_conn:.2f}")
            print(f"Q contains values: {set(self.Q.values())}")

    def sequence_from_string(self, seqstr):
        """Convert a string representation of a sequence to a list of 0/1 values."""
        return [int(c in 'h1') for c in seqstr.casefold() if not c.isspace()]

    def sequence_to_string(self):
        """Convert a list of 0/1 values to a string representation (H/P)."""
        return ''.join('H' if c == 1 else 'P' for c in self.sequence)

    def node_parity(self, node):
        """Return the parity (checkerboard coloring) of a lattice node."""
        return (node[0] % 2 + node[1] % 2) % 2

    def get_dict_HP_energy(self, graph, sequence):
        """Compute QUBO contributions for HP interactions."""
        Q_dict = defaultdict(float)
        sequence_len = len(sequence)
        for u, v in graph.edges():
            ev, od = (u, v) if self.node_parity(u) == 0 else (v, u)
            for i in range(sequence_len):
                for j in range(i + 1, sequence_len):
                    if abs(i - j) <= 2 or \
                        (i + j) % 2 == 0 or \
                            sequence[i] == 0 or \
                                sequence[j] == 0: 
                        continue
                    if i % 2 == 0:
                        Q_dict[((ev, i), (od, j))] += -1
                    else:
                        Q_dict[((ev, j), (od, i))] += -1
        return Q_dict

    def get_dict_unique_location(self, graph, strength, sequence_length):
        """Enforce that each bead occupies a unique lattice location."""
        Q_dict = defaultdict(float)
        for i in range(sequence_length):
            parity_i = i % 2
            for u in graph.nodes():
                if self.node_parity(u) == parity_i:
                    Q_dict[((u, i), (u, i))] += -strength
                    for v in graph.nodes():
                        if u != v and self.node_parity(v) == parity_i:
                            Q_dict[((u, i), (v, i))] += strength
        return Q_dict

    ## TODO: fix this one
    def get_dict_self_avoidance(self, graph, strength, sequence_length):
        """Enforce self-avoidance: beads cannot overlap."""
        Q_dict = defaultdict(float)
        evenpos = [i for i in range(sequence_length) if i % 2 == 0]
        oddpos = [i for i in range(sequence_length) if i % 2 == 1]
        for u in graph.nodes():
            positions = evenpos if self.node_parity(u) == 0 else oddpos
            for x in positions:
                for y in positions:
                    if x < y:
                        Q_dict[((u, x), (u, y))] += strength
        return Q_dict

    def get_dict_chain_connectivity(self, graph, strength, sequence_length):
        """Enforce chain connectivity: beads must be adjacent in the lattice."""
        Q_dict = defaultdict(float)
        last_even = sequence_length - 2 + sequence_length % 2
        last_odd = sequence_length - 1 - sequence_length % 2
        for i in range(sequence_length):
            for u in graph.nodes():
                for v in graph.nodes():
                    if u == v or (u, v) in graph.edges() or (v, u) in graph.edges():
                        continue
                    if self.node_parity(u) == 0 and self.node_parity(v) == 1:
                        if i % 2 == 0: # even
                            if i != last_even or sequence_length % 2 == 0:
                                Q_dict[((u, i), (v, i + 1))] += strength
                        else:           # odd
                            if i != last_odd or sequence_length % 2 == 1:
                                Q_dict[((u, i + 1), (v, i))] += strength                         
        return Q_dict


    def interaction_matrix(self):
        """Return the full QUBO interaction matrix."""
        return self.Q

    def optimization_matrix(self):
        """Return the QUBO matrix for HP optimization only."""
        return self.QHP

    def constraint_matrix_1(self):
        """Return the QUBO matrix for unique bead location constraint."""
        return self.Q1

    def constraint_matrix_2(self):
        """Return the QUBO matrix for self-avoidance constraint."""
        return self.Q2

    def constraint_matrix_3(self):
        """Return the QUBO matrix for chain connectivity constraint."""
        return self.Q3

    def Q_as_np_array(self, Q_dict):
        """Convert a QUBO dictionary to a numpy array."""
        n = len(self.keys)
        Q = np.zeros((n, n))
        for i, ki in enumerate(self.keys):
            for j, kj in enumerate(self.keys):
                Q[i, j] = Q_dict.get((ki, kj), 0.0)
        return Q

    def get_energies(self, bits):
        """Compute the energy contributions for a given bitstring."""
        qhp = 0.0
        q1 = self.lambd[0] * self.sequence_length
        q2 = 0.0
        q3 = 0.0
        for i, bi in enumerate(bits):
            if not bi:
                continue
            for j, bj in enumerate(bits):
                if not bj:
                    continue
                qhp += self.QHP.get((self.keys[i], self.keys[j]), 0.0)
                q1 += self.Q1.get((self.keys[i], self.keys[j]), 0.0)
                q2 += self.Q2.get((self.keys[i], self.keys[j]), 0.0)
                q3 += self.Q3.get((self.keys[i], self.keys[j]), 0.0)
        return qhp, q1, q2, q3

    def print_energies(self, bits):
        """Print the energy breakdown for a given bitstring."""
        qhp, q1, q2, q3 = self.get_energies(bits)
        total = qhp + q1 + q2 + q3
        print(f"EHP = {qhp}, E1 = {q1}, E2 = {q2}, E3 = {q3}, Total E = {total}")

    def to_ising(self):
        """Convert the QUBO interaction matrix to Ising form."""
        h_dict, J_dict, offset_ising = qubo_to_ising(self.interaction_matrix())
        return h_dict, J_dict, offset_ising

    def show_lattice(self, qubobitstring, axes=None, indices_fontsize=8, colors=None, rotation_angle=None):
        """Visualize the lattice and the protein configuration."""
        if axes is None:
            fig, axes = plt.subplots(figsize=(8, 8))

        # Checkerboard background
        lattice_dimensions = self.lattice_dimensions
        image = np.fromfunction(lambda i, j: self.node_parity([i, j]), lattice_dimensions, dtype=int)
        checkerboard_cmap = ListedColormap(["#eaeaea", "#fefefe"], name="lat_cmap")
        axes.matshow(image, cmap=checkerboard_cmap)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xticklabels([])
        axes.set_yticklabels([])

        # Amino acid colors
        colors = colors or ["#5878b7", "#cb3676"] # ["#11f033", "#f03311"]
        hp_cmap = ListedColormap(colors , name="hp_cmap") 
        fpos, xpos, ypos, posc = [], [], [], []
        xstart, ystart, cstart = [], [], []
        text_dict = defaultdict(list)

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
            text_dict[(s[0], s[1])].append(f)

        # Annotate lattice positions
        for (x, y), indices in text_dict.items():
            label = ",".join(map(str, indices))
            axes.text(x - 0.4, y - 0.3, label, color='k', fontsize=indices_fontsize, ha='left')

        # Sort positions by sequence index
        sorted_indices = np.argsort(fpos)
        xpos = np.array(xpos)[sorted_indices]
        ypos = np.array(ypos)[sorted_indices]
        posc = np.array(posc)[sorted_indices]

        axes.plot(xpos, ypos, 'k-', lw=0.5)
        axes.scatter(xpos, ypos, s=100, c=posc, cmap=hp_cmap, alpha=0.5)
        axes.scatter(xstart, ystart, c='k', s=25, marker=5)
