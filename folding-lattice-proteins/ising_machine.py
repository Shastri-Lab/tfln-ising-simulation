from tqdm import tqdm
import numpy as np
from numpy import sin, pi


ROOT2 = np.sqrt(2)
def sigma(x): # return np.tanh(ROOT2*x)
    return -0.5 + np.cos(pi/4 * (x-1))**2 # 0.5*sin(pi/2 * x) # equiv to: -1 + 2*np.cos(pi/4 * (x-1))**2

def solve_isingmachine(J, h, e_offset=0.0, target_energy=None, num_iterations=250_000, num_ics=2, alphas=None, betas=0.01, noise_std=0.1, early_break=True):
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
    repeat_factor = int(np.max(np.abs(b)) / np.max(np.abs(W))) # repeat the bias term to match the strength of the coupling term
    b /= repeat_factor # TODO: repeat factor might have a bad effect on different betas; should be ok if they aren't wildly different
    b = b.reshape(num_pars, num_spins, 1)
    W = np.concatenate([W]+[b]*repeat_factor, axis=-1) # shape (num_pars, num_spins, num_spins+repeat_factor)

    x_init = np.random.uniform(-0.25, 0.25, (num_ics, num_spins)) #-np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) # 
    x_vector = np.stack([x_init for _ in range(num_pars)]) # use the same initial state for all betas
    output = np.zeros_like(x_vector)
    noise = np.empty((num_pars, num_ics, num_spins))

    bits_history = []
    e_history = []

    print('Running simulation...')
    try:
        desc = f'target energy: {target_energy:.1f}' if target_energy else ''
        progress_bar = tqdm(range(num_iterations), dynamic_ncols=True, desc=desc)
        for t in progress_bar:
            spin_vector = np.sign(x_vector)     # σ ∈ {-1, 1}
            qubo_bits = (spin_vector+1)/2       # q ∈ {0, 1}
            
            current_energy = np.einsum('ijk,ijk->ij', spin_vector, np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h, spin_vector) + e_offset
            min_energy_idx_flat = np.argmin(current_energy)
            min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
            min_energy = current_energy[min_energy_idx]
            assert min_energy == current_energy.min()
            min_qubo_bits = qubo_bits[*min_energy_idx, :]

            # record the history
            bits_history.append(qubo_bits.astype(bool))
            e_history.append(current_energy.astype(np.float32))
            
            # break if we are close enough to the target energy
            if early_break and target_energy and np.any(np.abs(current_energy - target_energy) < 1e-3):
                break

            # compute the next state of the system
            noise[:] = np.random.normal(0, noise_std, (num_ics, num_spins))
            x_vector += noise
            np.einsum(
                'ijk,ihk->ihj',
                W, sigma(np.concatenate([x_vector]+[np.ones((num_pars, num_ics, 1))]*repeat_factor, axis=-1)),
                out=output)

            x_vector = output
            x_vector /= np.max(np.abs(x_vector), axis=-1, keepdims=True) # upload to AWG

            if target_energy:
                progress_bar.set_description(f"energy: {current_energy.min():.1f} / {target_energy:.1f}, num up: {int(np.sum(min_qubo_bits))}")
            else:
                progress_bar.set_description(f"energy: {current_energy.min():.1f}")
        print(f'Done.')
    except KeyboardInterrupt: # allow user to interrupt the simulation
        print(f'Interrupted.')
    print(f'Completed {t+1} iterations.')

    e_history = np.array(e_history)
    return x_vector, bits_history, e_history, alpha_beta, qubo_bits

