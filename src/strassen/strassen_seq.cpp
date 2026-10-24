#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static void matmul_reference(
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

static bool full_reference_check(
    const real_t* A,
    const real_t* B,
    const real_t* C,
    std::size_t n,
    double* rel_err
) {
    real_t* Cref = alloc_matrix(n);

    try {
        matmul_reference(Cref, A, B, n);
        // Strassen uses extra additions, so allow a slightly
        // looser tolerance than the naive kernels.
        const bool ok = verify_full(C, Cref, n, 1e-8, rel_err);
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

        for (int w = 0; w < args.warmup; ++w) {
            strassen_seq(C, A, B, n, args.cutoff);
        }

        bool all_verified = true;

        for (int run = 1; run <= args.runs; ++run) {
            const double t0 = now();
            strassen_seq(C, A, B, n, args.cutoff);
            const double elapsed = now() - t0;

            bool verified = false;
            double rel_err = 0.0;

            if (args.verify) {
                verified = verify_freivalds(
                    A, B, C, n, 1e-8, &rel_err
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
            info.algo = "strassen";
            info.model = "seq";
            info.n = n;
            info.P = 1;
            info.T = 1;
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
        std::cerr << "strassen_seq error: " << e.what() << '\n';
        std::free(A);
        std::free(B);
        std::free(C);
        return 2;
    }
}
