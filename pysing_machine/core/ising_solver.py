import logging
import numpy as np
from tqdm import tqdm
from numpy import sin, pi
from dataclasses import dataclass
from scipy.sparse import csr_matrix, issparse

@dataclass
class IsingProblem:
    J: np.ndarray
    h: np.ndarray
    e_offset: float = 0.0

@dataclass
class SolverConfig:
    target_energy: float = None
    num_iterations: int = 250_000
    num_ics: int = 2
    ic_range: float = 0.25
    alphas: np.ndarray = None
    betas: float = 0.01
    noise_std: float = 0.1
    early_break: bool = True
    simulated_annealing: bool = False
    annealing_iters: int = 1
    annealing_fraction: float = 1.0
    start_temperature: float = 10.0
    make_symmetric: bool = False
    sparse: bool = False
    bit_precision: int = None

@dataclass
class SovlerResults:
    final_vector: np.ndarray
    spin_bits_history: np.ndarray
    energy_history: np.ndarray
    success: bool


ROOT2 = np.sqrt(2)
def sigma(x): # return np.tanh(ROOT2*x)
    # return -1 + 2*np.cos(pi/4 * (x-1))**2 # 0.5*sin(pi/2 * x) # equiv to: -1 + 2*np.cos(pi/4 * (x-1))**2
    return sin(pi * x / 2)

def quadratic_matmul(J, x, sparse=False):
    # if issparse(J):
    #     return np.einsum(
    #         'ijk,ijk->ij',
    #         x,
    #         np.einsum(
    #             'ij,lmj->lmi',
    #             J.toarray(), x,
    #             ),
    #         )
    if sparse:
        out = np.zeros((x.shape[0], x.shape[1]))
        sparse_J = csr_matrix(J)
        for i in range(x.shape[0]): # batch dim (betas)
            for j in range(x.shape[1]): # ics dim
                out[i, j] = x[i, j].T @ sparse_J @ x[i, j]
        return out
    else:
        return np.einsum(
            'ijk,ijk->ij',
            x,
            np.einsum(
                'ij,lmj->lmi',
                J, x,
                ),
            )

def linear_matmul(h, x):
    return np.einsum(
        'k,ijk->ij',
        h, x,
        )

def calculate_energy(*pars, spins=None, sparse=False): # TODO: compare to energy calculated from X instead of sigma(X)
    energy = 0
    for J, h, e in pars:
        energy += quadratic_matmul(J, spins, sparse=sparse) + linear_matmul(h, spins) + e
    return energy

def get_param_grid(config: SolverConfig):
    alphas, betas = config.alphas, config.betas
    betas = np.atleast_1d(betas)
    if alphas is None:
        alphas = 1 - betas                    # alpha is complement of beta for running average
        alpha_beta = np.stack([alphas, betas], axis=-1).reshape(-1, 2)
    else:
        alphas = np.atleast_1d(alphas)        # allow for multiple alphas too
        alpha_beta = np.stack(np.meshgrid(alphas, betas), axis=-1).reshape(-1, 2)
    
    config.alpha_beta = alpha_beta

def quantize_matrix(M, bit_precision):
    """Quantize M to `bit_precision` bits with dynamic range scaling."""
    if bit_precision is None:
        return M # use full precision if bit_precision is None
    min_val, max_val = np.min(M), np.max(M)
    if max_val == min_val:  # avoid division by zero
        return M  # no quantization needed if all values are the same
    
    levels = 2 ** bit_precision
    M_scaled = (M - min_val) / (max_val - min_val)  # scale to [0, 1]
    M_discrete = np.round(M_scaled * levels) / levels  # quantize
    return min_val + M_discrete * (max_val - min_val)  # scale back

def initialize_problem(problem: IsingProblem, config: SolverConfig):
    J, h, e_offset = problem.J, problem.h, problem.e_offset

    if config.make_symmetric:
        J = (J + J.T)
    else:
        J = 2*J

    # get hyperparameters
    get_param_grid(config)
    assert J.shape[0] == J.shape[1], "J must be a square matrix"
    num_spins = J.shape[0]
    num_pars = config.alpha_beta.shape[0]
    num_ics = config.num_ics
    ic_range = config.ic_range

    # initialize iteration matrix
    W = np.zeros((num_pars, num_spins, num_spins))
    if h is not None:
        b = np.zeros((num_pars, num_spins))
    else:
        b = None
    for i, (alpha, beta) in enumerate(config.alpha_beta):
        W[i, :, :] = alpha * np.eye(num_spins) - beta * J / np.max(np.abs(J))  # normalize beta by the maximum coupling strength
        if h is not None:
            b[i, :] = -beta * h / np.max(np.abs(J))  # normalize beta by the maximum coupling strength
    
    # incorporate bias term into iteration matrix
    if b is not None:
        # in hardware, we have the issue that b is much bigger than W, so the bias term dominates the coupling term
        # to fix this, we scale it down so each element is the same size, then repeat it by the same factor so that the matmul result is unchanged
        repeat_factor = int(max(1.0, np.max(np.abs(b)) / np.max(np.abs(W)))) # repeat the bias term to match the strength of the coupling term
        print(f'Scaling down b and repeating {repeat_factor} times.')
        b /= repeat_factor # TODO: repeat factor might have a bad effect on different betas; should be ok if they aren't wildly different
        b = b.reshape(num_pars, num_spins, 1)
        W = np.concatenate([W]+[b]*repeat_factor, axis=-1) # shape (num_pars, num_spins, num_spins+repeat_factor)

        prepare_input = lambda x: np.concatenate([x]+[np.ones((num_pars, num_ics, 1))]*repeat_factor, axis=-1)
    else:
        prepare_input = lambda x: x

    # set up initial vectors
    x_init = np.random.uniform(-ic_range, ic_range, (num_ics, num_spins)) #-np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) # 
    x_vector = np.stack([x_init for _ in range(num_pars)]) # use the same initial state for all betas
    output = np.zeros_like(x_vector)
    noise = np.empty((num_pars, num_ics, num_spins))
    
    return num_spins, num_pars, W, x_vector, output, noise, prepare_input

def find_min_energy_index(current_energy, qubo_bits):
    min_energy_idx_flat = np.argmin(current_energy)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
    min_energy = current_energy[min_energy_idx]
    min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]
    return min_energy_idx, min_energy, min_qubo_bits

def update_state(W, x_in, output, sparse=False):
    num_spins = x_in.shape[2] # check if AI got this right
    
    if sparse: # probably not optimal to sparse-ify the matrix every iteration
        i, j, k = W.shape
        _, h, _ = x_in.shape
        for idx in range(i): # iterate over batch dimension (different betas)
            W_sparse = csr_matrix(W[idx])  # W[idx] is shape (j, k)
            sigma_x_in = sigma(x_in[idx])  # sigma_x_in is shape (h, k)
            output[idx] = W_sparse.dot(sigma_x_in.T).T # sparse matmul: result is shape (h, j)
    else:
        np.einsum(
            'ijk,ihk->ihj',
            W,
            sigma(x_in),
            out=output,
        )

    output /= np.max(np.abs(output), axis=-1, keepdims=True)

def solve_isingmachine(problem: IsingProblem, config: SolverConfig):
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
    success = None
    num_spins, num_pars, W, x_vector, output, noise, prepare_input = \
        initialize_problem(problem, config)
    W = quantize_matrix(W, config.bit_precision)
    x_vector = quantize_matrix(x_vector, config.bit_precision)
    output = quantize_matrix(output, config.bit_precision)

    # compute the energy of the initial state
    spin_vector = np.sign(x_vector)     # σ ∈ {-1, 1}
    qubo_bits = (spin_vector+1)/2       # q ∈ {0, 1}
    current_energy = calculate_energy((problem.J, problem.h, problem.e_offset), spins=spin_vector)
    min_energy_idx, min_energy, min_qubo_bits = find_min_energy_index(current_energy, qubo_bits)

    bits_history = [qubo_bits.astype(bool)] # save only the bits to save memory
    e_history = [current_energy.astype(np.float32)]
    
    std = config.noise_std
    delta_t = config.noise_std / config.num_iterations # TODO: add possibility for more annealing schedules

    print('Running simulation...')
    try:
        desc = f'target energy: {config.target_energy:.1f}' if config.target_energy else ''
        progress_bar = tqdm(range(config.num_iterations-1), dynamic_ncols=True, desc=desc)
        for t in progress_bar:
            if config.simulated_annealing:
                if t % config.annealing_iters == 0 and t != 0:
                    # std = max(0.01, std - delta_t)
                    # std /= 1.5 # TODO: annealing schedule config should use data class configuration too, this is weird.
                    std *= config.annealing_fraction

            # break if we are close enough to the target energy
            if config.early_break and config.target_energy and np.any(np.abs(current_energy - config.target_energy) < 1e-3):
                success = True
                break

            # compute the next state of the system
            noise[:] = np.random.normal(0, std, (config.num_ics, num_spins))
            update_state(W, prepare_input(x_vector+noise), output, sparse=config.sparse)
            x_vector = quantize_matrix(output, config.bit_precision)

            # record the history
            spin_vector = np.sign(x_vector)     # σ ∈ {-1, 1}
            qubo_bits = (spin_vector+1)/2       # q ∈ {0, 1}

            current_energy = calculate_energy((problem.J, problem.h, problem.e_offset), spins=spin_vector, sparse=config.sparse)
            e_history.append(current_energy.astype(np.float32))
            min_energy_idx, min_energy, min_qubo_bits = find_min_energy_index(current_energy, qubo_bits)
            bits_history.append(qubo_bits.astype(bool))
            

            if config.target_energy:
                progress_bar.set_description(f"energy: {current_energy.min():.1f} / {config.target_energy:.1f}, num up: {int(np.sum(min_qubo_bits))}, noise std: {std:.2f}")
            else:
                progress_bar.set_description(f"energy: {current_energy.min():.1f}")
        print(f'Done.')
    except KeyboardInterrupt: # allow user to interrupt the simulation
        print(f'Interrupted.')
    print(f'Completed {t+1} iterations.')

    if config.early_break and config.target_energy and success is None:
        success = False
    return SovlerResults(
        final_vector=x_vector,
        spin_bits_history=bits_history,
        energy_history=np.array(e_history),
        success=success
    )

import torch

def sigma_gpu(x): # return np.tanh(ROOT2*x)
    return -1 + 2*torch.cos(pi/4 * (x-1))**2 # 0.5*sin(pi/2 * x) # equiv to: -1 + 2*np.cos(pi/4 * (x-1))**2

def solve_isingmachine_gpu(
        J, h,
        e_offset=0.0,
        target_energy=None,
        num_iterations=250_000,
        num_ics=2,
        alphas=None,
        betas=0.01,
        noise_std=0.1,
        early_break=True,
        save_iter_freq=5,
        skip_energy_calculation=False
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
    
    """

    og_betas = np.atleast_1d(betas)
    betas = og_betas
    if alphas is None:
        alphas = 1 - betas                    # alpha is complement of beta for running average
        alpha_beta = np.stack([alphas, betas], axis=-1).reshape(-1, 2)
    else:
        alphas = np.atleast_1d(alphas)        # allow for multiple alphas too
        alpha_beta = np.stack(np.meshgrid(alphas, betas), axis=-1).reshape(-1, 2)

    num_spins = h.shape[0]
    num_pars = alpha_beta.shape[0]

    print('Converting into energy minimization problem...')
    W = np.zeros((num_pars, num_spins, num_spins))
    b = np.zeros((num_pars, num_spins))
    for i, (alpha, beta) in enumerate(alpha_beta):
        W[i, :, :] = alpha * np.eye(num_spins) - beta * J / np.max(np.abs(J))  # normalize beta by the maximum coupling strength
        b[i, :] = -beta * h / np.max(np.abs(J))  # normalize beta by the maximum coupling strength
    
    # in hardware, we have the issue that b is much bigger than W, so the bias term dominates the coupling term
    # to fix this, we scale it down so each element is the same size, then repeat it by the same factor so that the matmul result is unchanged
    repeat_factor = int(max(1.0, np.max(np.abs(b)) / np.max(np.abs(W)))) # repeat the bias term to match the strength of the coupling term
    print(f'Scaling down b and repeating {repeat_factor} times.')
    b /= repeat_factor # TODO: repeat factor might have a bad effect on different betas; should be ok if they aren't wildly different
    b = b.reshape(num_pars, num_spins, 1)
    W = np.concatenate([W]+[b]*repeat_factor, axis=-1) # shape (num_pars, num_spins, num_spins+repeat_factor)

    x_init = np.random.uniform(-0.25, 0.25, (num_ics, num_spins)) #-np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) # 
    x_vector = np.stack([x_init for _ in range(num_pars)]) # use the same initial state for all betas
    output = np.zeros_like(x_vector)
    noise = np.empty((num_pars, num_ics, num_spins))

    # tensorize - use torch from here onward
    x_vector = torch.tensor(x_vector, dtype=torch.float32).cuda()
    output = torch.tensor(output, dtype=torch.float32).cuda()
    noise = torch.tensor(noise, dtype=torch.float32).cuda()
    W = torch.tensor(W, dtype=torch.float32).cuda()
    if not skip_energy_calculation:
        J = torch.tensor(J, dtype=torch.float32).cuda()
        h = torch.tensor(h, dtype=torch.float32).cuda()

    # compute the energy of the initial state
    spin_vector = torch.sign(x_vector)     # σ ∈ {-1, 1}
    qubo_bits = (spin_vector + 1) / 2       # q ∈ {0, 1}
    bits_history = [qubo_bits.cpu().numpy().astype(bool)]

    if not skip_energy_calculation:
        current_energy = torch.einsum('ijk,ijk->ij', spin_vector, torch.einsum('ij,lmj->lmi', J, spin_vector)) + torch.einsum('k,ijk->ij', h, spin_vector) + e_offset
        e_history = [current_energy.cpu().numpy().astype(np.float32)]
    else:
        e_history = []

    print('Running simulation...')
    try:
        desc = f'target energy: {target_energy:.1f}' if target_energy else ''
        progress_bar = tqdm(range(num_iterations), dynamic_ncols=True, desc=desc)
        for t in progress_bar:
            # break if we are close enough to the target energy
            if early_break and target_energy and not skip_energy_calculation and torch.any(torch.abs(current_energy - target_energy) < 1e-3):
                break

            # compute the next state of the system
            noise[:] = torch.normal(0, noise_std, (num_ics, num_spins), device='cuda')
            output = torch.einsum(
                'ijk,ihk->ihj',
                W,
                sigma_gpu(torch.cat([x_vector+noise]+[torch.ones((num_pars, num_ics, 1), device='cuda')]*repeat_factor, axis=-1)),
                )
            output /= torch.max(torch.abs(output), axis=-1, keepdims=True)[0] # TODO: the [0] index was added because torch.max also returns the indices; torch.max also squeezes the dimensions so careful if any dim=1
            x_vector = output

            if t % save_iter_freq == 0: # do CPU stuff: record history, update progress bar
                spin_vector = torch.sign(x_vector)     # σ ∈ {-1, 1}
                qubo_bits = (spin_vector + 1) / 2       # q ∈ {0, 1}
                if not skip_energy_calculation:
                    current_energy = torch.einsum('ijk,ijk->ij', spin_vector, torch.einsum('ij,lmj->lmi', J, spin_vector)) + torch.einsum('k,ijk->ij', h, spin_vector) + e_offset
                    e_history.append(current_energy.cpu().numpy().astype(np.float32))
                bits_history.append(qubo_bits.cpu().numpy().astype(bool))
                if not skip_energy_calculation:
                    if target_energy:
                        progress_bar.set_description(f"energy: {current_energy.min().item():.1f} / {target_energy:.1f}")
                    else:
                        progress_bar.set_description(f"energy: {current_energy.min().item():.1f}")
                
        print(f'Done.')
    except KeyboardInterrupt: # allow user to interrupt the simulation
        print(f'Interrupted.')
    print(f'Completed {t+1} iterations.')

    e_history = np.array(e_history)
    return x_vector, bits_history, e_history, alpha_beta, qubo_bits

