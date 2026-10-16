#include "common.h"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

// Cấp phát bộ nhớ căn chỉnh 64 byte cho ma trận cục bộ.
// count là số phần tử double cần cấp phát.
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

// Tính tham chiếu ijk độc lập, chỉ chạy trên rank 0.
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

    // Không dùng MPI từ bên trong vùng OpenMP.
    MPI_Init_thread(
        &argc, &argv,
        MPI_THREAD_FUNNELED,
        &provided
    );

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int exit_code = 0;

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "MPI does not provide MPI_THREAD_FUNNELED\n";
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

    // MPI_Scatterv/Gatherv dùng số đếm và displacement kiểu int.
    if (n == 0 ||
        n > static_cast<std::size_t>(INT_MAX) ||
        n > static_cast<std::size_t>(INT_MAX) / n) {
        if (rank == 0) {
            std::cerr
                << "Matrix size is invalid or exceeds MPI int counts.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Chia n hàng cho các rank; các rank đầu nhận thêm hàng dư.
    std::vector<int> rows(world_size);
    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size);

    const int base_rows = static_cast<int>(n / world_size);
    const int remainder = static_cast<int>(n % world_size);

    int row_offset = 0;

    for (int r = 0; r < world_size; ++r) {
        rows[r] = base_rows + (r < remainder ? 1 : 0);
        send_counts[r] = rows[r] * static_cast<int>(n);
        displs[r] = row_offset * static_cast<int>(n);
        row_offset += rows[r];
    }

    const std::size_t local_rows =
        static_cast<std::size_t>(rows[rank]);
    const std::size_t local_elements = local_rows * n;

    // A và C đầy đủ chỉ cần trên rank 0.
    real_t* A = nullptr;
    real_t* C = nullptr;

    if (rank == 0) {
        A = alloc_matrix(n);
        C = alloc_matrix(n);
    }

    // B đầy đủ trên mọi rank sau khi broadcast.
    real_t* B = alloc_matrix(n);

    // Bộ đệm chứa các hàng A và C được giao cho rank hiện tại.
    real_t* local_A = alloc_aligned(local_elements);
    real_t* local_C = alloc_aligned(local_elements);

    const bool allocation_failed =
        (B == nullptr) ||
        (local_A == nullptr) ||
        (local_C == nullptr) ||
        (rank == 0 && (A == nullptr || C == nullptr));

    int local_alloc_error = allocation_failed ? 1 : 0;
    int any_alloc_error = 0;

    MPI_Allreduce(
        &local_alloc_error, &any_alloc_error,
        1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    if (any_alloc_error) {
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

    if (rank == 0) {
        gen_matrix(A, n, args.seed, 0, n);
        gen_matrix(B, n, args.seed + 1, 0, n);
    }

    // Truyền B đến các rank khác.
    MPI_Bcast(
        B, static_cast<int>(n * n),
        MPI_DOUBLE, 0, MPI_COMM_WORLD
    );

    // Chạy một lượt tính toán; trả về thời gian compute và communication.
    auto execute_once = [&](double* compute_time,
                            double* comm_time,
                            double* total_time) {
        double comm_local = 0.0;
        double compute_local = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        const double total_start = MPI_Wtime();

        // Scatter A.
        double comm_start = MPI_Wtime();

        MPI_Scatterv(
            A, send_counts.data(), displs.data(), MPI_DOUBLE,
            local_A, static_cast<int>(local_elements), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        comm_local += MPI_Wtime() - comm_start;

        // Tính C_local = A_local * B.
        const double compute_start = MPI_Wtime();

        if (local_rows > 0) {
            matmul_ikj_rows(
                local_C, local_A, B, local_rows, n
            );
        }

        compute_local = MPI_Wtime() - compute_start;

        // Gather C_local về rank 0.
        comm_start = MPI_Wtime();

        MPI_Gatherv(
            local_C, static_cast<int>(local_elements), MPI_DOUBLE,
            C, send_counts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        comm_local += MPI_Wtime() - comm_start;

        const double total_local = MPI_Wtime() - total_start;

        double max_compute = 0.0;
        double max_comm = 0.0;
        double max_total = 0.0;

        MPI_Reduce(
            &compute_local, &max_compute,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
        );

        MPI_Reduce(
            &comm_local, &max_comm,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
        );

        MPI_Reduce(
            &total_local, &max_total,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            *compute_time = max_compute;
            *comm_time = max_comm;
            *total_time = max_total;
        }
    };

    // Warm-up không được tính vào thời gian benchmark.
    for (int w = 0; w < args.warmup; ++w) {
        double tc = 0.0, tm = 0.0, tt = 0.0;
        execute_once(&tc, &tm, &tt);
    }

    for (int run = 1; run <= args.runs; ++run) {
        double t_compute = 0.0;
        double t_comm = 0.0;
        double t_total = 0.0;

        execute_once(&t_compute, &t_comm, &t_total);

        bool verified = false;
        double rel_err = 0.0;

        if (rank == 0 && args.verify) {
            verified = verify_freivalds(
                A, B, C, n, 1e-10, &rel_err
            );

            // Với n <= 1000, kiểm tra thêm bằng phép nhân ijk.
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
                std::cerr << "Verification failed for n="
                          << n << ", run=" << run << '\n';
                exit_code = 1;
            }
        }

        // Chỉ rank 0 ghi CSV để tránh nhiều process in trùng.
        if (rank == 0) {
            RunInfo info;
            info.algo = "naive_ikj";
            info.model = "mpi";
            info.n = n;
            info.P = world_size;
            info.T = 1;
            info.run = run;
            info.t_total = t_total;
            info.t_compute = t_compute;
            info.t_comm = t_comm;
            info.verified = args.verify ? verified : false;
            info.rel_err = rel_err;
            info.cutoff = 0;

            print_csv(info);
        }

        // Đồng bộ trạng thái lỗi giữa các rank.
        int global_error = 0;
        MPI_Allreduce(
            &exit_code, &global_error,
            1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
        );
        exit_code = global_error;
    }

    std::free(A);
    std::free(B);
    std::free(C);
    std::free(local_A);
    std::free(local_C);

    MPI_Finalize();
    return exit_code;
}