from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':
    solve_hp_problem(
    load_hp_model_by_name(
        'S6',
        latdim=(3,4),
        lambdas=(2.1, 2.4, 3.0),
        ),
    num_iterations=1000,
    num_ics=1500,
    alphas=None,
    betas=(0.005, 0.0075, 0.01, 0.025), 
    noise_std=0.2,
    is_plotting=True,
    is_saving=False,
    )
