from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':
    
    solve_hp_problem(
        load_hp_model_by_name(
            'S10',
            latdim=(4,3),
            lambdas=(2.1, 2.4, 3.0),
            ),
        num_iterations=2500,
        num_ics=5,
        alphas=(1),
        betas=(0.0005, 0.001 ,0.005,0.01, 0.05,0.1,0.5),
        noise_std=0.1,
        simulated_annealing=True,
        is_plotting=True,
        is_saving=False,
        )