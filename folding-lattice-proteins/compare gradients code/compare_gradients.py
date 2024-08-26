from tqdm import tqdm
import numpy as np
from numpy import sin, pi
import jax.numpy as npa
from jax import grad
import matplotlib.pyplot as plt
ROOT2 = np.sqrt(2)

def sigma(x):  # return np.tanh(ROOT2*x)
    return -1 + 2 * npa.cos(pi / 4 * (x - 1)) ** 2  # 0.5*sin(pi/2 * x) # equiv to: -1 + 2*np.cos(pi/4 * (x-1))**2
def sinpi(x):
    return np.sin(np.pi * x) * np.pi / 2.0

def cospi(x):
    return np.cos(np.pi/2 * x) * np.pi / 2.0

def compare_gradients(
        J, h,
        e_offset=0.0,
        target_energy=None,
        num_iterations=250_000,
        num_ics=2,
        alphas=None,
        betas=0.01,
        noise_std=0.1,
        early_break=True,
        simulated_annealing=False,
        desired_num_up=None,
):
    """
    Solve an Ising problem defined by J and h using the coherent Ising machine model.

    Args:
        J (np.ndarray): Coupling matrix.
        h (np.ndarray): Field vector.
        e_offset (float): Energy offset.
        target_energy (float): Target energy for early stopping.
        num_iterations (int): Number of iterations.
        num_ics (int): Number of initial conditions.
        alphas (np.ndarray): Decay rates.
        betas (np.ndarray): Learning rates.
        noise_std (float): Standard deviation of the noise.
        early_break (bool): Whether to stop early if the target energy is reached.
        simulated_annealing (bool): Whether to use simulated annealing acceptance criterion.

    """
    h *= 0
    # J *= 0
    # plt.matshow(J)
    # plt.matshow(J.transpose())
    # plt.show()

    def hamiltonian(x):
        spin_vector = sigma(x)
        return npa.sum(npa.einsum('ijk,ijk->ij', spin_vector,
                          npa.einsum('ij,lmj->lmi', J, spin_vector)) + npa.einsum('k,ijk->ij', h,
                                                                                spin_vector) + e_offset)
    og_betas = np.atleast_1d(betas)
    betas = og_betas
    if alphas is None:
        alphas = 1 - betas  # alpha is complement of beta for running average
        alpha_beta = np.stack([alphas, betas], axis=-1).reshape(-1, 2)
    else:
        alphas = np.atleast_1d(alphas)  # allow for multiple alphas too
        alpha_beta = np.stack(np.meshgrid(alphas, betas), axis=-1).reshape(-1, 2)
    num_spins = h.shape[0]
    num_pars = alpha_beta.shape[0]


    x_init = np.random.uniform(-0.25, 0.25, (
     num_ics,num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas

    noise = np.empty((num_pars,num_ics,num_spins))

    noise[:] = np.random.normal(0, noise_std, (num_pars,num_ics,num_spins))
    x_noise = x_vector + noise
    s_vector = sigma(x_vector)
    gradient = grad(hamiltonian)(x_vector)

    # out2 = np.einsum(
    #     'ij,lmj->lmi',
    #     J,
    #     s_vector,
    # )
    # out2 += np.einsum(
    #     'ij,lmj->lmi',
    #     J.transpose(),
    #     s_vector,
    # )

    out2 = np.einsum(
        'ij,lmj->lmi',
        J + J.transpose(),
        s_vector,
    )
    out3 = np.einsum(
        'k,lmk->lmk',
        h,
        cospi(x_vector),
    )
    gradient2 = (out2 * cospi(x_vector) + out3)
    gradient = np.sum(gradient[:,:,:],axis = 0)
    gradient2 = np.sum(gradient2[:,:,:],axis = 0)
    plt.subplot(1, 3, 1)
    plt.imshow(gradient)
    plt.colorbar(shrink = 0.8)
    plt.title("autodiff gradient")
    plt.subplot(1, 3, 2)
    plt.imshow(gradient2)
    plt.colorbar(shrink = 0.8)
    plt.title("analytic gradient")
    plt.subplot(1, 3, 3)
    plt.imshow(100 * np.abs(gradient2 - gradient) / np.abs(gradient))
    plt.colorbar(shrink = 0.8)
    plt.title("% Error")
    plt.tight_layout()
    plt.show()


def solve_isingmachine_works_1beta(
        J, h,
        e_offset=0.0,
        target_energy=None,
        num_iterations=250_000,
        num_ics=2,
        alphas=None,
        betas=0.01,
        noise_std=0.1,
        early_break=True,
        simulated_annealing=False,
        desired_num_up=None,
):
    """
    Solve an Ising problem defined by J and h using the coherent Ising machine model.

    Args:
        J (np.ndarray): Coupling matrix.
        h (np.ndarray): Field vector.
        e_offset (float): Energy offset.
        target_energy (float): Target energy for early stopping.
        num_iterations (int): Number of iterations.
        num_ics (int): Number of initial conditions.
        alphas (np.ndarray): Decay rates.
        betas (np.ndarray): Learning rates.
        noise_std (float): Standard deviation of the noise.
        early_break (bool): Whether to stop early if the target energy is reached.
        simulated_annealing (bool): Whether to use simulated annealing acceptance criterion.

    """
    # h *= 0
    # J *= 0
    # plt.matshow(J)
    # plt.matshow(J.transpose())
    # plt.show()

    def hamiltonian(x):
        spin_vector = sigma(x)
        # return npa.sum(npa.einsum('ijk,ijk->ij', spin_vector,
        #                   npa.einsum('ij,lmj->lmi', J, spin_vector)) + npa.einsum('k,ijk->ij', h,
        #                                                                         spin_vector) + e_offset)
        return npa.sum(npa.einsum('jk,jk->j', spin_vector,
                          npa.einsum('ij,mj->mi', J, spin_vector)) + npa.einsum('k,jk->j', h,
                                                                                spin_vector) + e_offset)


    og_betas = np.atleast_1d(betas)
    betas = og_betas
    if alphas is None:
        alphas = 1 - betas  # alpha is complement of beta for running average
        alpha_beta = np.stack([alphas, betas], axis=-1).reshape(-1, 2)
    else:
        alphas = np.atleast_1d(alphas)  # allow for multiple alphas too
        alpha_beta = np.stack(np.meshgrid(alphas, betas), axis=-1).reshape(-1, 2)
    num_spins = h.shape[0]
    num_pars = alpha_beta.shape[0]


    x_init = np.random.uniform(-0.25, 0.25, (
     num_ics,num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    # x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas
    x_vector = x_init.copy()
    noise = np.empty((num_ics,num_spins))

    noise[:] = np.random.normal(0, noise_std, (num_ics,num_spins))
    x_noise = x_vector + noise
    s_vector = sigma(x_vector)
    gradient = grad(hamiltonian)(x_vector)

    out2 = np.einsum(
        'ij,mj->mi',
        J,
        s_vector,
    )
    out2 += np.einsum(
        'ij,mj->mi',
        J.transpose(),
        s_vector,
    )

    out3 = np.einsum(
        'k,mk->mk',
        h,
        cospi(x_vector),
    )
    gradient2 = (out2 * cospi(x_vector) + out3)
    plt.subplot(1, 3, 1)
    plt.imshow(gradient)
    plt.colorbar(shrink = 0.8)
    plt.title("autodiff gradient")
    plt.subplot(1, 3, 2)
    plt.imshow(gradient2)
    plt.colorbar(shrink = 0.8)
    plt.title("analytic gradient")
    plt.subplot(1, 3, 3)
    plt.imshow(100 * np.abs(gradient2 - gradient) / np.maximum(np.abs(gradient), 1e-10))
    plt.colorbar(shrink = 0.8)
    plt.title("% Error")
    plt.show()

def solve_isingmachine_works_1beta_sigma(
        J, h,
        e_offset=0.0,
        target_energy=None,
        num_iterations=250_000,
        num_ics=2,
        alphas=None,
        betas=0.01,
        noise_std=0.1,
        early_break=True,
        simulated_annealing=False,
        desired_num_up=None,
):
    """
    Solve an Ising problem defined by J and h using the coherent Ising machine model.

    Args:
        J (np.ndarray): Coupling matrix.
        h (np.ndarray): Field vector.
        e_offset (float): Energy offset.
        target_energy (float): Target energy for early stopping.
        num_iterations (int): Number of iterations.
        num_ics (int): Number of initial conditions.
        alphas (np.ndarray): Decay rates.
        betas (np.ndarray): Learning rates.
        noise_std (float): Standard deviation of the noise.
        early_break (bool): Whether to stop early if the target energy is reached.
        simulated_annealing (bool): Whether to use simulated annealing acceptance criterion.

    """
    h *= 0
    # J *= 0
    # # plt.subplot(1,2,1)
    # plt.matshow(J)
    # # plt.subplot(1,2,2)
    # plt.matshow(J.transpose())
    # plt.show()

    def hamiltonian(spin_vector):
        # return npa.sum(npa.einsum('ijk,ijk->ij', spin_vector,
        #                   npa.einsum('ij,lmj->lmi', J, spin_vector)) + npa.einsum('k,ijk->ij', h,
        #                                                                         spin_vector) + e_offset)
        return npa.sum(npa.einsum('jk,jk->j', spin_vector,
                          npa.einsum('ij,mj->mi', J, spin_vector)) + npa.einsum('k,jk->j', h,
                                                                                spin_vector) + e_offset)


    og_betas = np.atleast_1d(betas)
    betas = og_betas
    if alphas is None:
        alphas = 1 - betas  # alpha is complement of beta for running average
        alpha_beta = np.stack([alphas, betas], axis=-1).reshape(-1, 2)
    else:
        alphas = np.atleast_1d(alphas)  # allow for multiple alphas too
        alpha_beta = np.stack(np.meshgrid(alphas, betas), axis=-1).reshape(-1, 2)
    num_spins = h.shape[0]
    num_pars = alpha_beta.shape[0]


    x_init = np.random.uniform(-0.25, 0.25, (
     num_ics,num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    # x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas
    x_vector = x_init.copy()
    noise = np.empty((num_ics,num_spins))

    noise[:] = np.random.normal(0, noise_std, (num_ics,num_spins))
    x_noise = x_vector + noise
    s_vector = sigma(x_vector)
    gradient = grad(hamiltonian)(s_vector)

    out2 = np.einsum(
        'ij,mj->mi',
        J,
        s_vector,
    )
    out2 += np.einsum(
        'ij,mj->mi',
        J.transpose(),
        s_vector,
    )

    # out3 = np.einsum(
    #     'k,k->k',
    #     h,
    #     cospi(x_noise),
    # )
    out3 = h
    gradient2 = out2 + out3
    # gradient = gradient[:,np.newaxis]
    # gradient2 = gradient2[:,np.newaxis]
    plt.subplot(1, 3, 1)
    plt.imshow(gradient)
    plt.colorbar(shrink = 0.8)
    plt.title("autodiff gradient")
    plt.subplot(1, 3, 2)
    plt.imshow(gradient2)
    plt.colorbar(shrink = 0.8)
    plt.title("analytic gradient")
    plt.subplot(1, 3, 3)
    plt.imshow(100 * np.abs(gradient2 - gradient) / np.maximum(np.abs(gradient), 1e-10))
    plt.colorbar(shrink = 0.8)
    plt.title("% Error")
    plt.show()

def solve_isingmachine_works_1ic_1beta(
        J, h,
        e_offset=0.0,
        target_energy=None,
        num_iterations=250_000,
        num_ics=2,
        alphas=None,
        betas=0.01,
        noise_std=0.1,
        early_break=True,
        simulated_annealing=False,
        desired_num_up=None,
):
    """
    Solve an Ising problem defined by J and h using the coherent Ising machine model.

    Args:
        J (np.ndarray): Coupling matrix.
        h (np.ndarray): Field vector.
        e_offset (float): Energy offset.
        target_energy (float): Target energy for early stopping.
        num_iterations (int): Number of iterations.
        num_ics (int): Number of initial conditions.
        alphas (np.ndarray): Decay rates.
        betas (np.ndarray): Learning rates.
        noise_std (float): Standard deviation of the noise.
        early_break (bool): Whether to stop early if the target energy is reached.
        simulated_annealing (bool): Whether to use simulated annealing acceptance criterion.

    """
    h *= 0
    # J *= 0
    # plt.subplot(1,2,1)
    plt.matshow(J)
    # plt.subplot(1,2,2)
    plt.matshow(J.transpose())
    plt.show()

    def hamiltonian(spin_vector):
        # return npa.sum(npa.einsum('ijk,ijk->ij', spin_vector,
        #                   npa.einsum('ij,lmj->lmi', J, spin_vector)) + npa.einsum('k,ijk->ij', h,
        #                                                                         spin_vector) + e_offset)
        # return npa.sum(npa.einsum('jk,jk->j', spin_vector,
        #                   npa.einsum('ij,mj->mi', J, spin_vector)) + npa.einsum('k,jk->j', h,
        #                                                                         spin_vector) + e_offset)

        return npa.sum(npa.einsum('k,k->', spin_vector,
                                  npa.einsum('ij,j->i', J, spin_vector)) + npa.einsum('k,k->', h,
                                                                                        spin_vector) + e_offset)

    og_betas = np.atleast_1d(betas)
    betas = og_betas
    if alphas is None:
        alphas = 1 - betas  # alpha is complement of beta for running average
        alpha_beta = np.stack([alphas, betas], axis=-1).reshape(-1, 2)
    else:
        alphas = np.atleast_1d(alphas)  # allow for multiple alphas too
        alpha_beta = np.stack(np.meshgrid(alphas, betas), axis=-1).reshape(-1, 2)
    num_spins = h.shape[0]
    num_pars = alpha_beta.shape[0]


    x_init = np.random.uniform(-0.25, 0.25, (
     num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    # x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas
    x_vector = x_init.copy()
    noise = np.empty((num_spins))

    noise[:] = np.random.normal(0, noise_std, (num_spins))
    x_noise = x_vector + noise
    s_vector = sigma(x_vector)
    gradient = grad(hamiltonian)(s_vector)

    out2 = np.einsum(
        'ij,j->i',
        J,
        s_vector,
    )
    out2 += np.einsum(
        'ij,j->i',
        J.transpose(),
        s_vector,
    )

    # out3 = np.einsum(
    #     'k,k->k',
    #     h,
    #     cospi(x_noise),
    # )
    out3 = h
    gradient2 = out2 + out3
    gradient = gradient[:,np.newaxis]
    gradient2 = gradient2[:,np.newaxis]
    plt.subplot(1, 3, 1)
    plt.imshow(gradient)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(gradient2)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(gradient2 - gradient)
    plt.colorbar()
    plt.show()

    plt.imshow(gradient2/gradient)
    plt.colorbar()
    plt.show()

def solve_isingmachine2(
        J, h,
        e_offset=0.0,
        target_energy=None,
        num_iterations=250_000,
        num_ics=2,
        alphas=None,
        betas=0.01,
        noise_std=0.1,
        early_break=True,
        simulated_annealing=False,
        desired_num_up=None,
):
    """
    Solve an Ising problem defined by J and h using the coherent Ising machine model.

    Args:
        J (np.ndarray): Coupling matrix.
        h (np.ndarray): Field vector.
        e_offset (float): Energy offset.
        target_energy (float): Target energy for early stopping.
        num_iterations (int): Number of iterations.
        num_ics (int): Number of initial conditions.
        alphas (np.ndarray): Decay rates.
        betas (np.ndarray): Learning rates.
        noise_std (float): Standard deviation of the noise.
        early_break (bool): Whether to stop early if the target energy is reached.
        simulated_annealing (bool): Whether to use simulated annealing acceptance criterion.

    """
    h *= 0;
    # J *=0
    plt.imshow(J)
    plt.show()
    def hamiltonian(x):
        spin_vector = sigma(x)
        # return npa.sum(npa.einsum('ijk,ijk->ij', spin_vector,
        #                   npa.einsum('ij,lmj->lmi', J, spin_vector)) + npa.einsum('k,ijk->ij', h,
        #                                                                         spin_vector) + e_offset)
        # return npa.sum(npa.einsum('jk,jk->j', spin_vector,
        #                   npa.einsum('ij,mj->mi', J, spin_vector)) + npa.einsum('k,jk->j', h,
        #                                                                         spin_vector) + e_offset)

        return npa.sum(npa.einsum('k,k->', spin_vector,
                                  npa.einsum('ij,j->i', J, spin_vector)) + npa.einsum('k,k->', h,
                                                                                        spin_vector) + e_offset)

    og_betas = np.atleast_1d(betas)
    betas = og_betas
    if alphas is None:
        alphas = 1 - betas  # alpha is complement of beta for running average
        alpha_beta = np.stack([alphas, betas], axis=-1).reshape(-1, 2)
    else:
        alphas = np.atleast_1d(alphas)  # allow for multiple alphas too
        alpha_beta = np.stack(np.meshgrid(alphas, betas), axis=-1).reshape(-1, 2)
    num_spins = h.shape[0]
    num_pars = alpha_beta.shape[0]


    x_init = np.random.uniform(-0.25, 0.25, (
     num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    # x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas
    x_vector = x_init.copy()
    noise = np.empty((num_spins))

    # compute the energy of the initial state
    spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}
    qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}

    bits_history = [qubo_bits.astype(bool)]

    noise[:] = np.random.normal(0, noise_std, (num_spins))
    x_noise = x_vector + noise
    gradient = grad(hamiltonian)(x_noise)

    # return npa.sum(npa.einsum('ijk,ijk->ij', spin_vector,
    #                           npa.einsum('ij,lmj->lmi', J, spin_vector)) + npa.einsum('k,ijk->ij', h,
    #                                                                                   spin_vector) + e_offset)
    out2 = np.einsum(
        'ji,j->i',
        J,
        sigma(x_noise),
    )

    out3 = np.einsum(
        'k,k->k',
        h,
        cospi(x_noise),
    )
    gradient2 =  out2 * cospi(x_noise) + out3
    gradient = gradient[:,np.newaxis]
    gradient2 = gradient2[:,np.newaxis]
    plt.subplot(1, 3, 1)
    plt.imshow(gradient)
    plt.colorbar(shrink = 0.5)
    plt.subplot(1, 3, 2)
    plt.imshow(gradient2)
    plt.colorbar(shrink = 0.5)
    plt.subplot(1, 3, 3)
    plt.imshow(gradient2 - gradient)
    plt.colorbar(shrink = 0.5)
    plt.show()
