from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':
    
    solve_hp_problem(
        load_hp_model_by_name(
            'S10',
            latdim=(4,3),
            lambdas=(2.1, 2.4, 3.0),
            ),
        num_iterations=2500,
        num_ics=25,
        alphas=None,
        betas=(0.01, 0.03,0.1,0.3,1.0,3.0),
        noise_std=0.1,
        simulated_annealing=True,
        is_plotting=True,
        is_saving=False,
        )