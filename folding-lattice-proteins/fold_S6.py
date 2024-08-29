from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':
    solve_hp_problem(
    load_hp_model_by_name(
        'S6',
        latdim=(3,4),
        lambdas=(2.1, 2.4, 3.0),
        ),
    num_iterations=2000,
    num_ics=10,
    alphas=None,
    betas=(0.005, 0.01,0.05,0.1,0.5),
    noise_std=0.1,
    is_plotting=True,
    is_saving=False,
    )

