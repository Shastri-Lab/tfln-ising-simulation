from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':    
    solve_hp_problem(
        load_hp_model_by_name(
            'S18',
            latdim=(5,5),
            lambdas=(2.1, 2.4, 3.0),
        ),
        num_iterations=10_000,
        num_ics=1000,
        # alphas=(1,), 
        betas=(0.01, 0.02, 0.03),
        noise_std=1.0,
        early_break=True,
        simulated_annealing=True,
        annealing_iters=200,
        annealing_fraction=0.95,
        is_plotting=True,
        is_saving=False,
        make_symmetric=True,
        sparse=False,
    )