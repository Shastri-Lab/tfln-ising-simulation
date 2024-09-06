from tqdm import tqdm
import numpy as np
from numpy import sin, pi
import jax.numpy as npa
from jax import grad
ROOT2 = np.sqrt(2)
import jax

def sigma(x):  # return np.tanh(ROOT2*x)
    return -1 + 2 * npa.cos(pi / 4 * (x - 1)) ** 2  # 0.5*sin(pi/2 * x) # equiv to: -1 + 2*np.cos(pi/4 * (x-1))**2
def sigma(x):  # return np.tanh(ROOT2*x)
    return npa.tanh(2 * x)


def solve_isingmachine_adam_Gibbs(
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
    Solve an Ising problem defined by J and h using automatic differentiation

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
    print(J.shape)

    @jax.jit
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

    b = np.zeros((num_pars, num_spins))
    for i, (alpha, beta) in enumerate(alpha_beta):
        b[i, :] = beta * np.ones_like(h)  # normalize beta by the maximum coupling strength


    x_init = np.random.uniform(-0.25, 0.25, (
    num_ics, num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas
    output = np.zeros_like(x_vector)
    noise = np.empty((num_pars, num_ics, num_spins))

    # compute the energy of the initial state
    spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}

    qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
    current_energy = np.einsum('ijk,ijk->ij', spin_vector, np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum(
        'k,ijk->ij', h, spin_vector) + e_offset
    min_energy_idx_flat = np.argmin(current_energy)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
    min_energy = current_energy[min_energy_idx]
    min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]

    bits_history = [qubo_bits.astype(bool)]
    e_history = []
    e_history.append(current_energy.astype(np.float32))

    temperature = 10.0

    mopt = np.zeros(x_vector.shape)
    vopt = np.zeros(x_vector.shape)

    xdum = np.zeros((1,1,h.shape[0]))
    moptdum = np.zeros_like(xdum)
    voptdum = np.zeros_like(xdum)
    bdum = np.zeros((1,h.shape[0]))
    noisesdum = np.zeros((1,num_spins,num_spins))

    @jax.jit
    def step_adam(gradient, mopt_old, vopt_old, iteration, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """ Performs one step of adam optimization"""
        mopt = beta1 * mopt_old + (1 - beta1) * gradient
        mopt_t = mopt / (1 - beta1 ** (iteration + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (npa.square(gradient))
        vopt_t = vopt / (1 - beta2 ** (iteration + 1))
        grad_adam = mopt_t / (npa.sqrt(vopt_t) + epsilon)
        return (grad_adam, mopt, vopt)
    @jax.jit
    def Gibbs(x_vector,mopt,vopt,iteration,b, noises):
        for k in range(x_vector.shape[-1]):
            noise = noises[:,:,k]
            # print(noises.shape)
            x_noise = x_vector + noise
            gradient = gradH(x_noise)
            (grad_adam, mopt_, vopt_) = step_adam(gradient[:, :, k], mopt[:, :, k], vopt[:, :, k],
                                                                  iteration)
            mopt = mopt.at[:,:,k].set(mopt_)
            vopt = vopt.at[:,:,k].set(vopt_)

            gradientscaled = npa.einsum(
                'i,ih->ih',
                b[:, k],
                grad_adam
            )
            x_vector_ = x_vector[:, :, k] - gradientscaled
            x_vector_ = npa.clip(x_vector_,a_min = -1, a_max = 1)

            x_vector = x_vector.at[:, :, k].set(x_vector_)
            # x_vector /= np.max(np.abs(x_vector), axis=-1, keepdims=True)
        return x_vector,mopt,vopt

    print('Running simulation...')
    iteration = 0
    gradH = jax.jit(grad(hamiltonian))
    # Gibbs(xdum, moptdum, voptdum, 0, bdum, noisesdum)
    # print("here")
    try:
        desc = f'target energy: {target_energy:.1f}' if target_energy else ''
        progress_bar = tqdm(range(num_iterations), dynamic_ncols=True, desc=desc)
        for t in progress_bar:
            # break if we are close enough to the target energy
            if early_break and target_energy and np.any(np.abs(current_energy - target_energy) < 1e-3):
                break
            # compute the next state of the system
            noises = np.random.normal(0, noise_std, (num_ics, num_spins,x_vector.shape[-1]))
            output,mopt,vopt = Gibbs(x_vector,mopt,vopt,iteration,b,noises)

            # compute the energy of the next state
            spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}
            qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
            current_energy = np.einsum('ijk,ijk->ij', spin_vector,
                                       np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h,
                                                                                             spin_vector) + e_offset


            min_energy_idx_flat = np.argmin(current_energy)
            min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
            min_energy = current_energy[min_energy_idx]
            min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]


            iteration += 1
            noise_std /= 1.01
            # record the history
            spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}
            qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
            last_energy = np.einsum('ijk,ijk->ij', spin_vector,
                                    np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h,
                                                                                          spin_vector) + e_offset
            e_history.append(last_energy.astype(np.float32))

            min_energy_idx_flat = np.argmin(last_energy)
            min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
            min_energy = current_energy[min_energy_idx]
            min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]
            bits_history.append(qubo_bits.astype(bool))

            if target_energy:
                progress_bar.set_description(
                    f"energy: {last_energy.min():.1f} / {target_energy:.1f}, num up: {int(np.sum(min_qubo_bits))}")
            else:
                progress_bar.set_description(f"energy: {last_energy.min():.1f}")
        print(f'Done.')
    except KeyboardInterrupt:  # allow user to interrupt the simulation
        print(f'Interrupted.')
    print(f'Completed {t + 1} iterations.')

    e_history = np.array(e_history)
    return x_vector, bits_history, e_history, alpha_beta, qubo_bits



def solve_isingmachine_adam(
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
    Solve an Ising problem defined by J and h using automatic differentiation

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
    print(J.shape)

    def hamiltonian(x):
        spin_vector = npa.tanh(5 * sigma(x))
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

    b = np.zeros((num_pars, num_spins))
    for i, (alpha, beta) in enumerate(alpha_beta):
        b[i, :] = beta * np.ones_like(h)  # normalize beta by the maximum coupling strength


    x_init = np.random.uniform(-0.25, 0.25, (
    num_ics, num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas
    output = np.zeros_like(x_vector)
    noise = np.empty((num_pars, num_ics, num_spins))

    # compute the energy of the initial state
    spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}

    qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
    current_energy = np.einsum('ijk,ijk->ij', spin_vector, np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum(
        'k,ijk->ij', h, spin_vector) + e_offset
    min_energy_idx_flat = np.argmin(current_energy)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
    min_energy = current_energy[min_energy_idx]
    min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]

    bits_history = [qubo_bits.astype(bool)]
    e_history = []
    e_history.append(current_energy.astype(np.float32))

    temperature = 10.0
    t_scale = 0.999
    last_energy = current_energy

    mopt = np.zeros(x_vector.shape)
    vopt = np.zeros(x_vector.shape)

    # @jax.jit
    def step_adam(gradient, mopt_old, vopt_old, iteration, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """ Performs one step of adam optimization"""
        mopt = beta1 * mopt_old + (1 - beta1) * gradient
        mopt_t = mopt / (1 - beta1 ** (iteration + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (npa.square(gradient))
        vopt_t = vopt / (1 - beta2 ** (iteration + 1))
        grad_adam = mopt_t / (npa.sqrt(vopt_t) + epsilon)
        return (grad_adam, mopt, vopt)
    print('Running simulation...')
    iteration = 0
    gradH = jax.jit(grad(hamiltonian))
    try:
        desc = f'target energy: {target_energy:.1f}' if target_energy else ''
        progress_bar = tqdm(range(num_iterations), dynamic_ncols=True, desc=desc)
        for t in progress_bar:
            # break if we are close enough to the target energy
            if early_break and target_energy and np.any(np.abs(current_energy - target_energy) < 1e-3):
                break
            # compute the next state of the system
            noise[:] = np.random.normal(0, noise_std, (num_ics, num_spins))
            if iteration % 200 == 0:
                noise_std /= 1.5
            x_noise = x_vector + noise

            gradient = gradH(x_noise)
            (grad_adam, mopt, vopt) = step_adam(gradient, mopt, vopt, iteration)
            iteration += 1
            gradientscaled = np.einsum(
                'ik,ihk->ihk',
                b,
                grad_adam
                )
            output = x_vector - gradientscaled

            output /= np.max(np.abs(output), axis=-1, keepdims=True)
            # np.clip(output, -1, 1, out=x_vector)

            # compute the energy of the next state
            spin_vector = np.sign(output)  # σ ∈ {-1, 1}
            qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
            current_energy = np.einsum('ijk,ijk->ij', spin_vector,
                                       np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h,
                                                                                             spin_vector) + e_offset

            min_energy_idx_flat = np.argmin(current_energy)
            min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
            min_energy = current_energy[min_energy_idx]
            min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]

            if simulated_annealing:
                # compare the current energy to the last energy
                delta_e = current_energy - last_energy
                acceptance_prob = np.exp(-delta_e / temperature)
                random_numbers = np.random.rand(*acceptance_prob.shape)
                accept = random_numbers < acceptance_prob
                x_vector[accept, :] = output[accept, :]
                temperature *= t_scale
                # print("DE",delta_e,acceptance_prob,temperature)

            else:
                x_vector = output

            # record the history
            spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}
            qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
            last_energy = np.einsum('ijk,ijk->ij', spin_vector,
                                    np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h,
                                                                                          spin_vector) + e_offset
            e_history.append(last_energy.astype(np.float32))

            min_energy_idx_flat = np.argmin(last_energy)
            min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
            min_energy = current_energy[min_energy_idx]
            min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]
            bits_history.append(qubo_bits.astype(bool))

            if target_energy:
                progress_bar.set_description(
                    f"energy: {last_energy.min():.1f} / {target_energy:.1f}, num up: {int(np.sum(min_qubo_bits))}")
            else:
                progress_bar.set_description(f"energy: {last_energy.min():.1f}")
        print(f'Done.')
    except KeyboardInterrupt:  # allow user to interrupt the simulation
        print(f'Interrupted.')
    print(f'Completed {t + 1} iterations.')

    e_history = np.array(e_history)
    return x_vector, bits_history, e_history, alpha_beta, qubo_bits



def solve_isingmachine(
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
    Solve an Ising problem defined by J and h using automatic differentiation

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
    print(J.shape)

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

    b = np.zeros((num_pars, num_spins))
    for i, (alpha, beta) in enumerate(alpha_beta):
        b[i, :] = beta * np.ones_like(h)  # normalize beta by the maximum coupling strength


    x_init = np.random.uniform(-0.25, 0.25, (
    num_ics, num_spins))  # -np.ones((num_ics, num_spins)) # np.zeros((num_ics, num_spins)) #
    x_vector = np.stack([x_init for _ in range(num_pars)])  # use the same initial state for all betas
    output = np.zeros_like(x_vector)
    noise = np.empty((num_pars, num_ics, num_spins))

    # compute the energy of the initial state
    spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}

    qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
    current_energy = np.einsum('ijk,ijk->ij', spin_vector, np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum(
        'k,ijk->ij', h, spin_vector) + e_offset
    min_energy_idx_flat = np.argmin(current_energy)
    min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
    min_energy = current_energy[min_energy_idx]
    min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]

    bits_history = [qubo_bits.astype(bool)]
    e_history = []
    e_history.append(current_energy.astype(np.float32))

    temperature = 10.0
    print('Running simulation...')
    try:
        desc = f'target energy: {target_energy:.1f}' if target_energy else ''
        progress_bar = tqdm(range(num_iterations), dynamic_ncols=True, desc=desc)
        for t in progress_bar:
            # break if we are close enough to the target energy
            if early_break and target_energy and np.any(np.abs(current_energy - target_energy) < 1e-3):
                break
            # compute the next state of the system
            noise[:] = np.random.normal(0, noise_std, (num_ics, num_spins))
            # noise_std /= 1.001
            x_noise = x_vector + noise
            gradient = grad(hamiltonian)(x_noise)

            gradientscaled = np.einsum(
                'ik,ihk->ihk',
                b,
                gradient
                )
            output = x_vector - gradientscaled

            output /= np.max(np.abs(output), axis=-1, keepdims=True)
            # np.clip(output, -1, 1, out=x_vector)

            # compute the energy of the next state
            spin_vector = np.sign(output)  # σ ∈ {-1, 1}
            qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
            current_energy = np.einsum('ijk,ijk->ij', spin_vector,
                                       np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h,
                                                                                             spin_vector) + e_offset

            min_energy_idx_flat = np.argmin(current_energy)
            min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
            min_energy = current_energy[min_energy_idx]
            min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]

            x_vector = output

            # record the history
            spin_vector = np.sign(x_vector)  # σ ∈ {-1, 1}
            qubo_bits = (spin_vector + 1) / 2  # q ∈ {0, 1}
            last_energy = np.einsum('ijk,ijk->ij', spin_vector,
                                    np.einsum('ij,lmj->lmi', J, spin_vector)) + np.einsum('k,ijk->ij', h,
                                                                                          spin_vector) + e_offset
            e_history.append(last_energy.astype(np.float32))

            min_energy_idx_flat = np.argmin(last_energy)
            min_energy_idx = np.unravel_index(min_energy_idx_flat, current_energy.shape)
            min_energy = current_energy[min_energy_idx]
            min_qubo_bits = qubo_bits[min_energy_idx[0], min_energy_idx[1], :]
            bits_history.append(qubo_bits.astype(bool))

            if target_energy:
                progress_bar.set_description(
                    f"energy: {last_energy.min():.1f} / {target_energy:.1f}, num up: {int(np.sum(min_qubo_bits))}")
            else:
                progress_bar.set_description(f"energy: {last_energy.min():.1f}")
        print(f'Done.')
    except KeyboardInterrupt:  # allow user to interrupt the simulation
        print(f'Interrupted.')
    print(f'Completed {t + 1} iterations.')

    e_history = np.array(e_history)
    return x_vector, bits_history, e_history, alpha_beta, qubo_bits
