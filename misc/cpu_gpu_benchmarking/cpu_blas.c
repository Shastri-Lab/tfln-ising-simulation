// cpu_blas.c
// This file implements a simple BLAS (Basic Linear Algebra Subprograms) function
// using the Accelerate framework on macOS. It provides a function to perform
// matrix-vector multiplication (dgemv) using double precision floating-point numbers.

/*
clang -O3 \                             # Optimization level 3 for maximum speed    
      -march=native \                   # Optimize for the current architecture
      -fPIC \                           # Position-independent code for shared libraries
      -shared \                         # Create a shared library
      -DACCELERATE_NEW_LAPACK \         # Use the modern Accelerate framework header
      -framework Accelerate \           # Link against the Accelerate framework
      -std=c11 \                        # Use the C11 standard
      -o libcpu_blas.so cpu_blas.c      # Output shared library name
*/

#define ACCELERATE_NEW_LAPACK 1         // Enable the new Accelerate LAPACK interface
#include <Accelerate/Accelerate.h>      // Include the Accelerate framework for BLAS functions

void dgemv_blas(const double *A, const double *x,
                double *y, size_t m, size_t n)
{
    const double alpha = 1.0, beta  = 0.0;

    /* stay LP32 → regular int */
    int M   = (int)m;
    int N   = (int)n;
    int lda = N;          /* row-major, so lda = n */

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                M, N,
                alpha, A, lda,
                x, 1,
                beta,  y, 1);
}
