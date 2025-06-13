#!/usr/bin/env python3
import ctypes, numpy as np, time, os, sys

def gflops(flops, t): return flops / t / 1e9

# ---- pure‑python NumPy fallback (no C involved) ----
def numpy_dgemv(A, x, y, m, n):
    """
    multiply A (m×n) by x (n) and store the result in y (m) – all numpy.
    we keep the signature consistent with the C wrappers so the bench()
    helper can treat every method uniformly.
    """
    np.dot(A, x, out=y)

# ---- helpers to wrap C funcs ----
def load(lib, sym, restype=None, argtypes=()):
    f = ctypes.cdll.LoadLibrary(lib).__getattr__(sym)
    f.restype, f.argtypes = restype, argtypes
    return f

pD = ctypes.POINTER(ctypes.c_double)
METHODS = {
    'numpy': numpy_dgemv,
    'blas': load('./libcpu_blas.so', 'dgemv_blas',
                 None, [pD, pD, pD, ctypes.c_size_t, ctypes.c_size_t]),
    'ref':  load('./libcpu_ref.so',  'dgemv_ref',
                 None, [pD, pD, pD, ctypes.c_size_t, ctypes.c_size_t]),
}

def bench(label, fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    t1 = time.perf_counter()
    return t1 - t0

def run_benchmarking_test(m, n, repeats=10_000, skip=False):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((m, n), dtype=np.float64)
    x = rng.standard_normal(n, dtype=np.float64)
    y = np.empty(m, dtype=np.float64)

    for method, fn in METHODS.items():
        output_file = f'results/bench_cpu_{method}_results_{n}.npz'
        if os.path.exists(output_file):
            if skip:
                print(f"Skipping {method} benchmark, {output_file} already exists.")
                continue
            else:
                print(f"Warning: {output_file} already exists. Overwriting.")

        # choose argument style based on implementation type
        if method == 'numpy':
            warm_args = (A, x, y, m, n)
        else:
            warm_args = (A.ctypes.data_as(pD), x.ctypes.data_as(pD),
                         y.ctypes.data_as(pD), m, n)

        # warm‑up
        bench(method, fn, *warm_args)

        deltas = np.zeros(repeats, dtype=np.float64)
        for i in range(repeats):
            deltas[i] = bench(method, fn, *warm_args)

        # save results
        np.savez(output_file, deltas=deltas)
        print(f"[{method}] results saved to {output_file}")

        avg_time = np.mean(deltas)
        std_time = np.std(deltas)
        min_time = np.min(deltas)
        max_time = np.max(deltas)
        print(f"[{method}] min: {min_time:.5f}s, max: {max_time:.5f}s")
        print(f"[{method}] mean: {avg_time:.5f}s ± {std_time:.5f}s (std)")
        flops = 2 * m * n
        print(f"[{method}] GFLOPS: {gflops(flops, avg_time):.2f} ± "
              f"{gflops(flops, std_time):.2f} (std)\n")


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'sweep':
            # Sweep over a range of sizes
            for m in [2**i for i in range(4, 14)]:
                print(f"Running benchmark for m = n = {m}")
                run_benchmarking_test(m, m, skip=True)
        else:
            m = int(sys.argv[1])
            n = int(sys.argv[2]) if len(sys.argv) > 2 else m
            print(f"Running benchmark for m = {m}, n = {n}")
            run_benchmarking_test(m, n)
    else:
        # Default case: run a single test with m = n = 1024
        m = n = 1024
        print(f"Running benchmark for m = n = {m}")
        run_benchmarking_test(m, n)

# Example usage:
# python benchmark_cpu.py sweep
# python benchmark_cpu.py 1024
# python benchmark_cpu.py 512 256
