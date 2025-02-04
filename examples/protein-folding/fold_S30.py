from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':    
    solve_hp_problem(
        load_hp_model_by_name(
            'S30',
            latdim=(6,7),
            lambdas=(2.1, 2.4, 3.0),
        ),
        num_iterations=1_000,
        num_ics=5_000,
        alphas=(1,), 
        betas=(0.01, 0.05),
        noise_std=1.0,
        early_break=True,
        simulated_annealing=True,
        is_plotting=True,
        is_saving=False,
        make_symmetric=True,
        sparse=True,
    )