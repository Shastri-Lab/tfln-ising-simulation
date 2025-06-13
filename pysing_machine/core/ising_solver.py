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
    alphas: np.ndarray = None
    betas: float = 0.01
    num_ics: int = 1
    ic_range: float = 0.5
    num_iterations: int = 10_000
    
    early_break: bool = True
    target_energy: float = None
    
    annealing_schedule: str = 'exponential'
    annealing_rate: float = 0.1
    custom_schedule: callable = None
    start_temperature: float = 1.0
    
    make_symmetric: bool = False
    sparse: bool = False
    bit_precision: int = None

@dataclass
class SolverResults:
    final_vector: np.ndarray
    spin_bits_history: np.ndarray
    energy_history: np.ndarray
    success: np.ndarray


ROOT2 = np.sqrt(2)
def sigma(x): 
    # return -1 + 2*np.cos(pi/4 * (x-1))**2 # 0.5*sin(pi/2 * x) # equiv to: -1 + 2*np.cos(pi/4 * (x-1))**2
    return np.tanh(pi * x / 2) # slope matches sin(pi * x / 2) at x=0, but has no problems when x is large
    # return sin(pi * x / 2)

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
    if sparse: # TODO: double loop is super slow
        out = np.zeros((x.shape[0], x.shape[1]))
        sparse_J = csr_matrix(J) # TODO: we shouldn't be doing this every iteration, we should have this already sparse if the flag is set
        for i in range(x.shape[0]): # batch dim (betas)
            for j in range(x.shape[1]): # ics dim
                out[i, j] = x[i, j].T @ sparse_J @ x[i, j]
        return out
    else:
        # return np.einsum('ij,ihj,ihj->ih', J, x, x) # this is apparently an optimization of the below? Doesn't work, but let's figure out later
        return np.einsum( # TODO: dense einsum is inefficient: np.einsum('ij,lmj->lmi', J, x) is unnecessary when J is dense, x @ J @ x.T would be faster.
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
    for J, h, e in pars: # TODO: the looping here is weird, we should be able to do this in one go?
        energy += quadratic_matmul(J, spins, sparse=sparse)
        if h is not None:
            energy += linear_matmul(h, spins)
        energy += e
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

def quantize_matrix(M, bit_precision, axis, min_val=None, max_val=None):
    
    """Quantize M to `bit_precision` bits with dynamic range scaling.
    
    If M has shape (num_parameters, N, N), each (N, N) matrix is quantized separately.
    If M has shape (num_parallel_runs, num_parameters, N), each (1, N) row is quantized separately.
    """
    # breakpoint() # currently working without this, so we shoudln't get into this function
    if bit_precision is None:
        return M # use full precision if bit_precision is None

    min_arr_val = np.min(M, axis=axis, keepdims=True)
    max_arr_val = np.max(M, axis=axis, keepdims=True)
    min_val = min_val * np.ones_like(min_val) if min_val is not None else min_arr_val
    max_val = max_val * np.ones_like(max_val) if max_val is not None else max_arr_val

    # Avoid division by zero where min_val == max_val
    no_range_mask = (max_val == min_val)

    levels = 2 ** bit_precision - 1 # minus 1 because we want to include the min and max values

    # Scale, quantize, and re-scale back, only for non-zero-range values
    M_scaled = np.where(no_range_mask, M, (M - min_val) / (max_val - min_val))
    M_discrete = np.round(M_scaled * levels) / levels
    M_quantized = np.where(no_range_mask, M, min_val + M_discrete * (max_val - min_val))

    return M_quantized

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

    J_max = np.max(np.abs(J))
    if J_max == 0:
        raise ValueError("J has all zeros, cannot normalize.")
    for i, (alpha, beta) in enumerate(config.alpha_beta):
        W[i, :, :] = alpha * np.eye(num_spins) - beta * J / J_max  # normalize beta by the maximum coupling strength
        if h is not None:
            b[i, :] = -beta * h / J_max  # normalize beta by the maximum coupling strength
    
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

def update_state(W, x_in, output, sparse=False, bit_precision=None):
    num_spins = x_in.shape[2] # check if AI got this right
    
    if sparse: # probably not optimal to sparse-ify the matrix every iteration
        # i, j, k = W.shape
        # _, h, _ = x_in.shape
        # for idx in range(i): # iterate over batch dimension (different betas)
        #     W_sparse = csr_matrix(W[idx])  # W[idx] is shape (j, k)
        #     sigma_x_in = sigma(x_in[idx])  # sigma_x_in is shape (h, k)
        #     output[idx] = W_sparse.dot(sigma_x_in.T).T # sparse matmul: result is shape (h, j)
        for i in range(W.shape[0]): # TODO: check that this optimization produces the same result
            output[i] = csr_matrix(W[i]) @ sigma(x_in[i]).T
    elif bit_precision is not None:
        np.einsum(
            'ijk,ihk->ihj',
            W,
            quantize_matrix(sigma(x_in), bit_precision, (2,), max_val=1, min_val=-1),
            out=output,
        )
    else:
        np.einsum(
            'ijk,ihk->ihj',
            W,
            sigma(x_in),
            out=output,
        )

    output /= np.max(np.abs(output), axis=-1, keepdims=True)

def get_annealing_temperature(t, config):
    if config.annealing_schedule == None or config.annealing_schedule == "constant":
        return config.start_temperature
    elif config.annealing_schedule == "exponential":
        return config.start_temperature * np.exp(-config.annealing_rate * t)
    elif config.annealing_schedule == "linear":
        return max(0.01, config.start_temperature - config.annealing_rate * t)
    elif config.annealing_schedule == "logarithmic":
        return config.start_temperature / (1 + config.annealing_rate * np.log(1 + t))
    elif config.annealing_schedule == "custom" and config.custom_schedule is not None:
        return config.custom_schedule(t)
    else:
        raise ValueError(f"Unknown annealing schedule: {config.annealing_schedule}")

def solve_isingmachine(problem: IsingProblem, config: SolverConfig):
    """
    Solve an Ising problem defined by J and h using the coherent Ising machine model.

    Args:
        problem (IsingProblem): A dataclass containing the definition of the Ising problem via a J matrix and an h vector.
        config (SolverConfig): A dataclass containing all the configuration options for the core Ising machine solver.
    
    """
    success = None
    num_spins, num_pars, W, x_vector, output, noise, prepare_input = \
        initialize_problem(problem, config)
    if config.bit_precision is not None: # TODO: not really necessary because it's handled in quantize_matrix
        W = quantize_matrix(W, config.bit_precision, (1,2), max_val=1, min_val=-1)
        x_vector = quantize_matrix(x_vector, config.bit_precision, (2,), max_val=1, min_val=-1)
        output = quantize_matrix(output, config.bit_precision, (2,), max_val=1, min_val=-1)

    # compute the energy of the initial state
    spin_vector = np.sign(x_vector)     # σ ∈ {-1, 1}
    qubo_bits = ((spin_vector+1)/2).astype(bool)       # q ∈ {0, 1}
    current_energy = calculate_energy((problem.J, problem.h, problem.e_offset), spins=spin_vector)

    bits_history = [qubo_bits] # save only the bits to save memory
    e_history = [current_energy.astype(np.float32)]

    # initialize a matrix to store success flag for each of the parallel runs
    success = np.zeros_like(current_energy, dtype=bool)
    target_energy = config.target_energy or -np.inf

    try:
        desc = f'target energy: {config.target_energy:.1f}' if config.target_energy else ''
        progress_bar = tqdm(range(config.num_iterations-1), dynamic_ncols=True, desc=desc)
        for t in progress_bar:
            std = get_annealing_temperature(t, config)

            # check if each of the parallel runs have already reached the target energy
            success = np.logical_or(success, np.abs(current_energy - target_energy) < 1e-5) # TODO: it's really important that the provided target energy is correct!! for the number partitioning, we first scale the set but we forget to scale the target energy, so the energy turns out to be wrong

            # break if we are close enough to the target energy
            if config.early_break and np.any(success):
                break

            # compute the next state of the system
            noise[:] = np.random.normal(0, std, (config.num_ics, num_spins))
            update_state(W, prepare_input(x_vector+noise), output, sparse=config.sparse, bit_precision=config.bit_precision)
            if config.bit_precision is not None: # TODO: not really necessary because it's handled in quantize_matrixs
                x_vector = quantize_matrix(output, config.bit_precision, (2,), max_val=1, min_val=-1)
            else:
                x_vector = np.clip(output, -1, 1) # TODO: I think this can be removed because update_state normalizes the output to max(abs(output))=1


            # record the history
            spin_vector = np.sign(x_vector)                 # σ ∈ {-1, 1}
            qubo_bits = ((spin_vector+1)/2).astype(bool)    # q ∈ {0, 1}

            current_energy = calculate_energy((problem.J, problem.h, problem.e_offset), spins=spin_vector, sparse=config.sparse)
            e_history.append(current_energy.astype(np.float32))
            bits_history.append(qubo_bits)
            
            if t % 1 == 0: # update progress bar every 10 iterations
                _, _, min_qubo_bits = find_min_energy_index(current_energy, qubo_bits)
                update_str = f"energy: {current_energy.min():.1f}"

            if config.target_energy:
                    update_str += f" | target: {config.target_energy:.1f}"
                    update_str += f" | noise std: {std:.2f}"
                    if t > 0:
                        num_changed_bits = np.sum(qubo_bits ^ bits_history[-2], axis=-1)
                        num_changed_bits_mean = np.mean(num_changed_bits)
                        num_changed_bits_std = np.std(num_changed_bits)
                        update_str += f" | changed: {num_changed_bits_mean:.2f} ± {num_changed_bits_std:.2f} / {num_spins}"
                    update_str += debug_text
                progress_bar.set_description(update_str)
                
                
    except KeyboardInterrupt: # allow user to interrupt the simulation
        print(f'Interrupted.')
    print(f'Completed {t+1} iterations.')

    return SolverResults(
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

