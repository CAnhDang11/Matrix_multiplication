#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <omp.h>

// ------------------------------------------------------------
// OpenMP ijk kernel
// Parallelize rows of C using schedule(static).
// Each thread writes to distinct rows of C.
// ------------------------------------------------------------
static void matmul_ijk_omp(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n,
    int threads
) {
    if (!C || !A || !B || n == 0 || threads <= 0) {
        throw std::invalid_argument("matmul_ijk_omp: invalid input");
    }

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;

            for (std::size_t k = 0; k < n; ++k) {
                sum += A[static_cast<std::size_t>(i) * n + k]
                     * B[k * n + j];
            }

            C[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }
}

// ------------------------------------------------------------
// OpenMP ikj kernel
// Parallelize rows of C. The inner loop accesses B and C
// contiguously, improving spatial locality.
// ------------------------------------------------------------
static void matmul_ikj_omp(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n,
    int threads
) {
    if (!C || !A || !B || n == 0 || threads <= 0) {
        throw std::invalid_argument("matmul_ikj_omp: invalid input");
    }

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const std::size_t row = static_cast<std::size_t>(i);
        real_t* crow = C + row * n;
        const real_t* arow = A + row * n;

        std::fill(crow, crow + n, 0.0);

        for (std::size_t k = 0; k < n; ++k) {
            const real_t aik = arow[k];
            const real_t* brow = B + k * n;

            for (std::size_t j = 0; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

// ------------------------------------------------------------
// Full reference verification using the sequential ijk kernel.
// ------------------------------------------------------------
static bool full_reference_check(
    const real_t* A,
    const real_t* B,
    const real_t* C,
    std::size_t n,
    double* rel_err
) {
    real_t* Cref = alloc_matrix(n);

    try {
        matmul_ijk(Cref, A, B, n);
        const bool ok = verify_full(C, Cref, n, 1e-10, rel_err);
        std::free(Cref);
        return ok;
    } catch (...) {
        std::free(Cref);
        throw;
    }
}

int main(int argc, char** argv) {
    real_t* A = nullptr;
    real_t* B = nullptr;
    real_t* C = nullptr;

    try {
        const Args args = parse_args(argc, argv);
        const std::size_t n = args.n;

        A = alloc_matrix(n);
        B = alloc_matrix(n);
        C = alloc_matrix(n);

        gen_matrix(A, n, args.seed, 0, n);
        gen_matrix(B, n, args.seed + 1, 0, n);

        const bool use_ijk = (args.kernel == 1);

        auto kernel = [&](real_t* out) {
            if (use_ijk) {
                matmul_ijk_omp(out, A, B, n, args.T);
            } else {
                matmul_ikj_omp(out, A, B, n, args.T);
            }
        };

        // Warmup runs are excluded from timing.
        for (int w = 0; w < args.warmup; ++w) {
            kernel(C);
        }

        bool all_verified = true;

        for (int run = 1; run <= args.runs; ++run) {
            const double t0 = now();
            kernel(C);
            const double elapsed = now() - t0;

            bool verified = false;
            double rel_err = 0.0;

            if (args.verify) {
                verified = verify_freivalds(
                    A, B, C, n, 1e-10, &rel_err
                );

                if (n <= 1000) {
                    double full_err = 0.0;
                    const bool full_ok = full_reference_check(
                        A, B, C, n, &full_err
                    );

                    rel_err = std::max(rel_err, full_err);
                    verified = verified && full_ok;
                }

                all_verified = all_verified && verified;
            }

            RunInfo info;
            info.algo = use_ijk ? "naive_ijk" : "naive_ikj";
            info.model = "omp";
            info.n = n;
            info.P = 1;
            info.T = args.T;
            info.run = run;
            info.t_total = elapsed;
            info.t_compute = elapsed;
            info.t_comm = 0.0;
            info.verified = verified;
            info.rel_err = rel_err;
            info.cutoff = args.cutoff;

            print_csv(info);

            if (args.verify && !verified) {
                std::cerr
                    << "Verification FAILED: n=" << n
                    << ", threads=" << args.T
                    << ", run=" << run
                    << ", rel_err=" << rel_err << '\n';
            }
        }

        std::free(A);
        std::free(B);
        std::free(C);

        if (args.verify && !all_verified) {
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "naive_omp error: " << e.what() << '\n';

        std::free(A);
        std::free(B);
        std::free(C);

        return 2;
    }
}
