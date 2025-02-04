from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':    
    solve_hp_problem(
        load_hp_model_by_name(
            'S10',
            latdim=(4,3),
            lambdas=(2.1, 2.4, 3.0),
        ),
        num_iterations=10_000,
        num_ics=1000,
        alphas=(1,), 
        betas=(0.05, 0.1, 0.3),
        noise_std=0.01,
        early_break=True,
        simulated_annealing=False,
        is_plotting=True,
        is_saving=False,
        make_symmetric=True,
        sparse=True,
    )