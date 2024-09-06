from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':
    solve_hp_problem(
        load_hp_model_by_name(
            'S24',
            latdim=(10, 10),
            lambdas=(2.1, 2.4, 3.0),
        ),
        num_iterations=10000,
        num_ics=255,
        alphas=None,
        betas=(0.3, 1, 3, 10),
        noise_std=0.05,
        simulated_annealing=True,
        is_plotting=True,
        is_saving=False,
    )