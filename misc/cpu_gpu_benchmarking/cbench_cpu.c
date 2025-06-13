/* cbench_cpu.c
   build:
     clang -O3 -march=native -std=c11 -fPIC -framework Accelerate \
           -o cbench_cpu cbench_cpu.c
   run:
     ./cbench_cpu            # benchmarks all sizes in SIZES[]
     ./cbench_cpu 4096 1000  # custom: m=n=4096, 1000 repeats
*/

#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

static inline double tod() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#if defined(__APPLE__)
  #include <mach/mach_time.h>
  static inline double now_sec(void)
  {
      static mach_timebase_info_data_t tb;
      if (tb.denom == 0) mach_timebase_info(&tb);
      return mach_absolute_time() * (double)tb.numer / tb.denom * 1e-9;
  }
#else
  #include <time.h>
  static inline double now_sec(void)
  {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
      return ts.tv_sec + ts.tv_nsec * 1e-9;
  }
#endif



/* ----------------- accelerate / cblas wrapper --------------------------- */
void dgemv_blas(const double *A, const double *x,
                double *y, size_t m, size_t n)
{
    const double alpha = 1.0, beta = 0.0;
    int M = (int)m, N = (int)n, lda = N;     /* row-major, so lda = n */
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                M, N, alpha, A, lda, x, 1, beta, y, 1);
}

/* --------------------------- helpers ------------------------------------ */
static void *xmalloc(size_t sz) {
    void *p = malloc(sz);
    if (!p) { perror("malloc"); exit(EXIT_FAILURE); }
    return p;
}

static void rand_fill(double *buf, size_t n) {
    for (size_t i = 0; i < n; ++i)
        buf[i] = drand48() - 0.5;
}

static void save_raw(const char *fn, const double *data, size_t n) {
    FILE *f = fopen(fn, "wb");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }
    fwrite(data, sizeof(double), n, f);
    fclose(f);
}

/* ---------------------------- benchmark ---------------------------------- */
typedef void (*matvec)(const double*, const double*, double*, size_t, size_t);

static void run_one(matvec fn, const char *tag,
                    const double *A, const double *x, double *y,
                    size_t m, size_t n, size_t reps)
{
    char fn_out[128];
    snprintf(fn_out, sizeof fn_out,
             "results/cbench_cpu_%s_results_%zu.bin", tag, n);

    double *deltas = xmalloc(reps * sizeof *deltas);

    /* warm-up */
    fn(A, x, y, m, n);

    for (size_t i = 0; i < reps; ++i) {
        // double t0 = tod();
        double t0 = now_sec();
        fn(A, x, y, m, n);
        // double t1 = tod();
        double t1 = now_sec();
        deltas[i] = t1 - t0;
    }

    save_raw(fn_out, deltas, reps);
    
    /* terse stats */
    double sum = 0.0, best = deltas[0], worst = deltas[0];
    for (size_t i = 0; i < reps; ++i) {
        sum += deltas[i];
        if (deltas[i] < best)  best  = deltas[i];
        if (deltas[i] > worst) worst = deltas[i];
    }
    double mean = sum / reps;
    double gflops = (2.0 * m * n) / (mean * 1e9);
    printf("[%s] n=%zu  best %.6fs  worst %.6fs  mean %.6fs  %.2f GFLOP/s\n",
        tag, n, best, worst, mean, gflops);
    free(deltas);
}

/* default sweep sizes (powers of two 16-8192) */
static const size_t SIZES[] = {
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
};

int main(int argc, char **argv)
{
    size_t reps = 10000, m, n;

    if (argc >= 3) {                 /* custom size+reps */
        m = n = strtoull(argv[1], NULL, 10);
        reps = strtoull(argv[2], NULL, 10);
    }

    srand48(0);                      /* reproducible */

    size_t n_sizes = (argc >= 3) ? 1 : sizeof SIZES / sizeof *SIZES;
    for (size_t s = 0; s < n_sizes; ++s) {
        if (argc < 3) m = n = SIZES[s];

        size_t mn = m * n, bytes = mn * sizeof(double);
        double *A = xmalloc(bytes);
        double *x = xmalloc(n * sizeof(double));
        double *y = xmalloc(m * sizeof(double));

        rand_fill(A, mn);
        rand_fill(x, n);

        run_one(dgemv_blas, "blas", A, x, y, m, n, reps);

        free(A); free(x); free(y);
    }
    return 0;
}