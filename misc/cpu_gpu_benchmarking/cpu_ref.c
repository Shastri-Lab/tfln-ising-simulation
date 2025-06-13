// cpu_ref.c
// This file implements a simple reference implementation of the
// Basic Linear Algebra Subprograms (BLAS) function dgemv (double precision
// general matrix-vector multiplication) using C. It provides a function
// to perform matrix-vector multiplication (dgemv) using double precision
// floating-point numbers. This implementation is intended for benchmarking
// purposes and is purposely written to be as slow as possible to
// provide a baseline for performance comparisons against optimized libraries
// such as BLAS (Basic Linear Algebra Subprograms).

/*
clang -O0 \                         # Optimization level 0 because I want it to go as slow as possible
      -fPIC \                       # Position-independent code for shared libraries
      -shared \                     # Create a shared library
      -std=c11 \                    # Use the C11 standard
      -o libcpu_ref.so cpu_ref.c    # Output shared library name

*/

#include <stddef.h>

void dgemv_ref(const double *A, const double *x,
               double *y, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < n; ++j)
            acc += A[i*n + j] * x[j];
        y[i] = acc;
    }
}