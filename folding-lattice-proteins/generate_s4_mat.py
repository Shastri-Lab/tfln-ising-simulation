from hp_lattice import *
from dimod.utilities import qubo_to_ising
from dwave.samplers import SimulatedAnnealingSampler
from dwave_to_isingmachine import save_qubo_model_to_ising_mat

print("Generating QUBO model...")
model = Lattice_HP_QUBO(
    dim = [3, 3],
    sequence="HPPH",
    Lambda=(2.1, 2.4, 3.0),
)

print("Converting QUBO to Ising...")
Q_qubo = model.interaction_matrix()
h_ising, J_ising, offset_ising = qubo_to_ising(Q_qubo)

print("Saving MATLAB file as 's4_3x3.mat'...", end=" ")
save_qubo_model_to_ising_mat(model, 's4_3x3.mat')
print("done.")