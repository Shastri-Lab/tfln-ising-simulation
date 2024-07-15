function e = hp_ising_energy(spins, J, h, offset, lambdas, sequence)
    e = spins * J * spins' + h * spins' + offset + lambdas(1) * length(sequence);
end