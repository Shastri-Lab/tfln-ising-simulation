import cupy as cp
import time
import os
import numpy as np

def run_benchmarking_test(M, N, repeats=10_000, skip=False):
    rng = cp.random.default_rng(0)
    A = rng.standard_normal((M, N), dtype=cp.float64)
    x = rng.standard_normal(N, dtype=cp.float64)

    output_file=f'data/bench_gpu_cupy_results_{N}.npz'
    if os.path.exists(output_file):
        if skip:
            print(f'Skipping CuPy benchmark, {output_file} already exists.')
            return
        else:
            print(f'Warning: {output_file} already exists. Overwriting.')

    # warm up gpu (avoid lazy init timing)
    _ = A @ x

    deltas = np.zeros(repeats, dtype=np.float64)

    for i in range(repeats):
        start = time.perf_counter()
        y = A @ x
        cp.cuda.Device(0).synchronize() # ensure complete
        end = time.perf_counter()
        deltas[i] = end-start

    np.savez(output_file, deltas=deltas)
    print(f'[CuPy]: results saved to {output_file}')
    
    avg_time = deltas.mean()
    std_time = deltas.std()
    min_time = deltas.min()
    max_time = deltas.max()
    print(f'[CuPy] min: {min_time:.5f}s, max: {max_time:.5f}s')
    print(f'[CuPy] mean: {avg_time:.5f}s, std: {std_time:.5f}s')


if __name__ == '__main__':
    for m in [2**i for i in range(4, 14)]:
        print(f'Running benchmark for m = n = {m}')
        run_benchmarking_test(m, m, skip=True)
