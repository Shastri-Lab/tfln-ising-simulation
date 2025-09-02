from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':    
    solve_hp_problem(
        load_hp_model_by_name(
            'S10',
            latdim=(4,4),
            lambdas=(2.1, 2.4, 3.0),
        ),
        num_iterations=10_000,
        num_ics=1000,
        alphas=(1,), 
        betas=(0.025, 0.05, 0.075),
        early_break=True,
        simulated_annealing=False,
        is_plotting=True,
        is_saving=False,
        make_symmetric=True,
        sparse=False,

        noise_std=1.0,
        annealing_iters=1,
        annealing_fraction=0.9997,

    )