#include "common.h"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

real_t* alloc_aligned(std::size_t count) {
    if (count == 0) {
        count = 1;
    }

    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, count * sizeof(real_t)) != 0) {
        return nullptr;
    }

    return static_cast<real_t*>(ptr);
}

// Mỗi OpenMP thread tính các hàng C khác nhau,
// do đó không có race condition khi ghi vào C.
void matmul_ikj_omp(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t nrows,
    std::size_t n,
    int threads
) {
    if (nrows == 0) {
        return;
    }

    std::fill(C, C + nrows * n, 0.0);

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (long long i = 0; i < static_cast<long long>(nrows); ++i) {
        const real_t* arow = A + i * n;
        real_t* crow = C + i * n;

        for (std::size_t k = 0; k < n; ++k) {
            const double aik = arow[k];
            const real_t* brow = B + k * n;

            for (std::size_t j = 0; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

// Phép nhân tham chiếu độc lập, chỉ chạy trên rank 0.
void matmul_reference(
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

} // namespace

int main(int argc, char** argv) {
    int provided = 0;

    MPI_Init_thread(
        &argc, &argv,
        MPI_THREAD_FUNNELED,
        &provided
    );

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "MPI_THREAD_FUNNELED is not supported.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Args args;

    try {
        args = parse_args(argc, argv);
    } catch (...) {
        if (rank == 0) {
            std::cerr << "Failed to parse command-line arguments.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const std::size_t n = args.n;

    if (n == 0 ||
        args.T < 1 ||
        args.runs < 1 ||
        args.warmup < 0 ||
        n > static_cast<std::size_t>(INT_MAX) ||
        n > static_cast<std::size_t>(INT_MAX) / n) {
        if (rank == 0) {
            std::cerr << "Invalid matrix size or runtime arguments.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Tắt điều chỉnh số thread tự động để benchmark ổn định hơn.
    omp_set_dynamic(0);

    // Chia hàng không đều: các rank đầu nhận thêm một hàng nếu có dư.
    std::vector<int> rows(world_size);
    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size);

    const int base_rows = static_cast<int>(n / world_size);
    const int remainder = static_cast<int>(n % world_size);

    int row_offset = 0;

    for (int r = 0; r < world_size; ++r) {
        rows[r] = base_rows + (r < remainder ? 1 : 0);
        counts[r] = rows[r] * static_cast<int>(n);
        displs[r] = row_offset * static_cast<int>(n);
        row_offset += rows[r];
    }

    const std::size_t local_rows =
        static_cast<std::size_t>(rows[rank]);
    const std::size_t local_elements = local_rows * n;

    real_t* A = nullptr;
    real_t* C = nullptr;

    if (rank == 0) {
        A = alloc_matrix(n);
        C = alloc_matrix(n);
    }

    // B được broadcast cho mỗi rank trong từng lượt chạy.
    real_t* B = alloc_aligned(n * n);
    real_t* local_A = alloc_aligned(local_elements);
    real_t* local_C = alloc_aligned(local_elements);

    int allocation_failed =
        (B == nullptr || local_A == nullptr || local_C == nullptr ||
         (rank == 0 && (A == nullptr || C == nullptr))) ? 1 : 0;

    int any_allocation_failed = 0;

    MPI_Allreduce(
        &allocation_failed, &any_allocation_failed,
        1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    if (any_allocation_failed) {
        if (rank == 0) {
            std::cerr << "Matrix allocation failed.\n";
        }

        std::free(A);
        std::free(B);
        std::free(C);
        std::free(local_A);
        std::free(local_C);

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Chỉ rank 0 khởi tạo dữ liệu toàn cục.
    if (rank == 0) {
        gen_matrix(A, n, args.seed, 0, n);
        gen_matrix(B, n, args.seed + 1, 0, n);
    }

    // Thực hiện một lượt: phân phối dữ liệu, tính toán, thu kết quả.
    auto execute_once = [&](double* out_total,
                            double* out_compute,
                            double* out_comm) {
        double local_comm = 0.0;
        double local_compute = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        const double total_start = MPI_Wtime();

        // 1. Scatter các hàng của A.
        double comm_start = MPI_Wtime();

        MPI_Scatterv(
            A, counts.data(), displs.data(), MPI_DOUBLE,
            local_A, static_cast<int>(local_elements), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        local_comm += MPI_Wtime() - comm_start;

        // 2. Broadcast B, tính cả thời gian truyền vào t_comm.
        comm_start = MPI_Wtime();

        MPI_Bcast(
            B, static_cast<int>(n * n),
            MPI_DOUBLE, 0, MPI_COMM_WORLD
        );

        local_comm += MPI_Wtime() - comm_start;

        // 3. Nhân ma trận cục bộ bằng OpenMP.
        const double compute_start = MPI_Wtime();

        matmul_ikj_omp(
            local_C, local_A, B,
            local_rows, n, args.T
        );

        local_compute = MPI_Wtime() - compute_start;

        // 4. Thu các hàng C về rank 0.
        comm_start = MPI_Wtime();

        MPI_Gatherv(
            local_C, static_cast<int>(local_elements), MPI_DOUBLE,
            C, counts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        local_comm += MPI_Wtime() - comm_start;

        const double local_total = MPI_Wtime() - total_start;

        // Lấy thời gian lớn nhất trong các rank.
        double max_total = 0.0;
        double max_compute = 0.0;
        double max_comm = 0.0;

        MPI_Reduce(
            &local_total, &max_total,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
        );

        MPI_Reduce(
            &local_compute, &max_compute,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
        );

        MPI_Reduce(
            &local_comm, &max_comm,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            *out_total = max_total;
            *out_compute = max_compute;
            *out_comm = max_comm;
        }
    };

    // Warm-up: không ghi vào kết quả benchmark.
    for (int w = 0; w < args.warmup; ++w) {
        double tt = 0.0, tc = 0.0, tm = 0.0;
        execute_once(&tt, &tc, &tm);
    }

    int exit_code = 0;

    for (int run = 1; run <= args.runs; ++run) {
        double t_total = 0.0;
        double t_compute = 0.0;
        double t_comm = 0.0;

        execute_once(&t_total, &t_compute, &t_comm);

        int verify_failed = 0;
        bool verified = false;
        double rel_err = 0.0;

        if (rank == 0 && args.verify) {
            verified = verify_freivalds(
                A, B, C, n, 1e-10, &rel_err
            );

            if (n <= 1000) {
                real_t* Cref = alloc_matrix(n);

                if (Cref == nullptr) {
                    std::cerr << "Reference allocation failed.\n";
                    verified = false;
                } else {
                    matmul_reference(Cref, A, B, n);

                    double full_err = 0.0;
                    const bool full_ok = verify_full(
                        C, Cref, n, 1e-10, &full_err
                    );

                    rel_err = std::max(rel_err, full_err);
                    verified = verified && full_ok;

                    std::free(Cref);
                }
            }

            if (!verified) {
                verify_failed = 1;
                std::cerr << "Verification failed: n="
                          << n << ", run=" << run << '\n';
            }
        }

        int global_verify_failed = 0;

        MPI_Allreduce(
            &verify_failed, &global_verify_failed,
            1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
        );

        if (global_verify_failed) {
            exit_code = 1;
        }

        if (rank == 0) {
            RunInfo info;
            info.algo = "naive_ikj";
            info.model = "hybrid";
            info.n = n;
            info.P = world_size;
            info.T = args.T;
            info.run = run;
            info.t_total = t_total;
            info.t_compute = t_compute;
            info.t_comm = t_comm;
            info.verified = args.verify ? verified : false;
            info.rel_err = rel_err;
            info.cutoff = 0;

            print_csv(info);
        }
    }

    std::free(A);
    std::free(B);
    std::free(C);
    std::free(local_A);
    std::free(local_C);

    MPI_Finalize();
    return exit_code;
}