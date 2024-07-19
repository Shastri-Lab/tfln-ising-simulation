import cProfile
import pstats
import numpy as np
from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':
    is_profiling = False

    if is_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    solve_hp_problem(
        load_hp_model_by_name(
            'S10',
            latdim=(4,3),
            lambdas=(2.1, 2.4, 3.0),
            ),
        num_iterations=1000,
        num_ics=10000,
        alphas=None, #(0.1, 0.3, 0.5, 0.8, 0.9, 1.0), # 0.85, 0.9, 0.999), # np.logspace(0, -0.25, 5),
        betas=(0.0025, 0.005, 0.0075), # (0.001, 0.0025, 0.005, 0.01, 0.1), # 0.0025, 0.05, 0.1), # np.logspace(-4, -0.25, 5),
        noise_std=0.25,
        is_plotting=True,
        is_saving=False,
        simulated_annealing=True,
        separate_energies=False,
        adjust_mean=False,
        )
    
    if is_profiling:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)