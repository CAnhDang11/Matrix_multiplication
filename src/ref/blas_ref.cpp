#include "common.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

// Full O(n^3) reference check for small matrices.
static void compute_ijk(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n
) {
    std::fill(C, C + n * n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;

            for (std::size_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }

            C[i * n + j] = sum;
        }
    }
}

static void blas_matmul(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n
) {
    if (n > static_cast<std::size_t>(__INT_MAX__)) {
        throw std::invalid_argument("Matrix dimension exceeds BLAS int limit");
    }

    const int N = static_cast<int>(n);

    // C = A * B, row-major, double precision.
    // beta = 0 means the previous contents of C are ignored.
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, N, N,
        1.0,
        A, N,
        B, N,
        0.0,
        C, N
    );
}

int main(int argc, char** argv) {
    real_t* A = nullptr;
    real_t* B = nullptr;
    real_t* C = nullptr;
    real_t* Cref = nullptr;

    try {
        const Args args = parse_args(argc, argv);
        const std::size_t n = args.n;

        A = alloc_matrix(n);
        B = alloc_matrix(n);
        C = alloc_matrix(n);

        // Reproducible inputs: B uses seed + 1.
        gen_matrix(A, n, args.seed, 0, n);
        gen_matrix(B, n, args.seed + 1, 0, n);

        // Warmup runs are excluded from timing.
        for (int i = 0; i < args.warmup; ++i) {
            blas_matmul(C, A, B, n);
        }

        bool all_verified = true;
        double worst_error = 0.0;

        for (int run = 1; run <= args.runs; ++run) {
            const double t0 = now();
            blas_matmul(C, A, B, n);
            const double elapsed = now() - t0;

            bool verified = false;
            double rel_err = 0.0;

            if (args.verify) {
                // Freivalds check is O(n^2) and works at all sizes.
                verified = verify_freivalds(
                    A, B, C, n, 1e-10, &rel_err
                );

                // Full reference check for n <= 1000.
                if (n <= 1000) {
                    if (Cref == nullptr) {
                        Cref = alloc_matrix(n);
                    }

                    compute_ijk(Cref, A, B, n);

                    double full_error = 0.0;
                    const bool full_ok = verify_full(
                        C, Cref, n, 1e-10, &full_error
                    );

                    rel_err = std::max(rel_err, full_error);
                    verified = verified && full_ok;
                }

                all_verified = all_verified && verified;
                worst_error = std::max(worst_error, rel_err);
            }

            RunInfo info;
            info.algo = "blas";
            info.model = "shared";
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
                std::cerr << "Verification failed on run "
                          << run << ", relative error = "
                          << rel_err << '\n';
            }
        }

        std::free(A);
        std::free(B);
        std::free(C);
        std::free(Cref);

        if (args.verify && !all_verified) {
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "blas_ref error: " << e.what() << '\n';

        std::free(A);
        std::free(B);
        std::free(C);
        std::free(Cref);

        return 2;
    }
}

// mkdir -p bin
// OPENBLAS_PREFIX="$(brew --prefix openblas)"
// g++-16 -std=c++17 -O3 -march=native -fopenmp -Iinclude -I"$OPENBLAS_PREFIX/include" src/common/common.cpp src/ref/blas_ref.cpp -L"$OPENBLAS_PREFIX/lib" -Wl,-rpath,"$OPENBLAS_PREFIX/lib" -lopenblas -o bin/blas_ref
// ./bin/blas_ref -n 1000 -t 4 -r 3 --verify
