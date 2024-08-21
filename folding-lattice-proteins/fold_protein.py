import cProfile
import pstats
import numpy as np
from ising_protein_folding import load_hp_model_by_name, solve_hp_problem
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Solve HP protein folding problem')
    parser.add_argument('--model', default='S30', help='HP model name')
    parser.add_argument('--latdim', nargs=2, type=int, default=[6, 7], help='Lattice dimensions')
    parser.add_argument('--lambdas', nargs=3, type=float, default=[2.1, 2.4, 3.0], help='Lambda values')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--num_ics', type=int, default=10000, help='Number of initial conditions')
    parser.add_argument('--alphas', nargs='*', type=float, help='Alpha values')
    parser.add_argument('--betas', nargs='*', type=float, default=[0.0025, 0.005, 0.0075], help='Beta values')
    parser.add_argument('--noise_std', type=float, default=0.25, help='Noise standard deviation')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--save', action='store_true', help='Enable saving')
    parser.add_argument('--simulated_annealing', action='store_true', help='Enable simulated annealing')
    parser.add_argument('--separate_energies', action='store_true', help='Use separate energies')
    parser.add_argument('--adjust_mean', action='store_true', help='Adjust mean')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    solve_hp_problem(
        load_hp_model_by_name(
            args.model,
            latdim=tuple(args.latdim),
            lambdas=tuple(args.lambdas),
            ),
        num_iterations=args.num_iterations,
        num_ics=args.num_ics,
        alphas=args.alphas,
        betas=args.betas,
        noise_std=args.noise_std,
        is_plotting=args.plot,
        is_saving=args.save,
        simulated_annealing=args.simulated_annealing,
        separate_energies=args.separate_energies,
        adjust_mean=args.adjust_mean,
        )   
    
    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)
