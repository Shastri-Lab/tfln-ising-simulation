from ising_protein_folding import load_hp_model_by_name, solve_hp_problem

if __name__ == '__main__':
    solve_hp_problem(
    load_hp_model_by_name(
        'S4',
        latdim=(3,4),
        lambdas=(2.1, 2.4, 3.0),
        ),
    num_iterations=10000,
    num_ics=1,
    alphas=(1), #(0.1, 0.3, 0.5, 0.8, 0.9, 1.0), # 0.85, 0.9,  0.999), # np.logspace(0, -0.25, 5),
    # betas=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5), # 0.0025, 0.05, 0.1), # np.logspace(-4, -0.25, 5),
    betas = (0.01,),
    noise_std=0.0,
    is_plotting=True,
    is_saving=False,
    )
