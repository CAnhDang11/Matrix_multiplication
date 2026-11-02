// strassen_hybrid.cpp — TV3 · D4
// Strassen lai ghep MPI + OpenMP: Rank 0 phan ra A, B thanh 7 cap toan hang,
// phan phoi round-robin; moi worker dung strassen_omp() de tinh M_i da luong.
// Thu gom M_i ve rank 0 va ghep thanh C.
//
// Bien dich:
//   mpicxx -std=c++17 -O3 -march=native -fopenmp -Iinclude \
//     src/common/common.cpp \
//     src/strassen/strassen.cpp \
//     src/strassen/strassen_hybrid.cpp \
//     -o bin/strassen_hybrid

#include "common.h"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

// Cap phat bo dem can 64 byte.
real_t* alloc_buf(std::size_t count) {
    if (count == 0) count = 1;
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = std::malloc(count * sizeof(real_t));
#else
    if (posix_memalign(&ptr, 64, count * sizeof(real_t)) != 0) return nullptr;
#endif
    return static_cast<real_t*>(ptr);
}

void mat_add(real_t* C, const real_t* A, const real_t* B, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) C[i] = A[i] + B[i];
}

void mat_sub(real_t* C, const real_t* A, const real_t* B, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) C[i] = A[i] - B[i];
}

void split4(const real_t* A, std::size_t n,
            real_t* A11, real_t* A12, real_t* A21, real_t* A22) {
    const std::size_t m = n / 2;
    for (std::size_t i = 0; i < m; ++i) {
        std::memcpy(A11 + i * m, A +  i      * n,     m * sizeof(real_t));
        std::memcpy(A12 + i * m, A +  i      * n + m, m * sizeof(real_t));
        std::memcpy(A21 + i * m, A + (i + m) * n,     m * sizeof(real_t));
        std::memcpy(A22 + i * m, A + (i + m) * n + m, m * sizeof(real_t));
    }
}

void join4(real_t* C, std::size_t n,
           const real_t* C11, const real_t* C12,
           const real_t* C21, const real_t* C22) {
    const std::size_t m = n / 2;
    for (std::size_t i = 0; i < m; ++i) {
        std::memcpy(C +  i      * n,     C11 + i * m, m * sizeof(real_t));
        std::memcpy(C +  i      * n + m, C12 + i * m, m * sizeof(real_t));
        std::memcpy(C + (i + m) * n,     C21 + i * m, m * sizeof(real_t));
        std::memcpy(C + (i + m) * n + m, C22 + i * m, m * sizeof(real_t));
    }
}

// Tinh tham chieu ijk
void matmul_ref(real_t* C, const real_t* A, const real_t* B, std::size_t n) {
    std::fill(C, C + n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

struct Task {
    std::vector<real_t> L;
    std::vector<real_t> R;
    std::vector<real_t> M;
};

std::vector<Task> build_tasks(
    const real_t* A11, const real_t* A12,
    const real_t* A21, const real_t* A22,
    const real_t* B11, const real_t* B12,
    const real_t* B21, const real_t* B22,
    std::size_t m)
{
    const std::size_t m2 = m * m;
    std::vector<Task> tasks(7);
    for (auto& t : tasks) { t.L.resize(m2); t.R.resize(m2); t.M.resize(m2, 0.0); }

    mat_add(tasks[0].L.data(), A11, A22, m2); mat_add(tasks[0].R.data(), B11, B22, m2);
    mat_add(tasks[1].L.data(), A21, A22, m2); std::memcpy(tasks[1].R.data(), B11, m2 * sizeof(real_t));
    std::memcpy(tasks[2].L.data(), A11, m2 * sizeof(real_t)); mat_sub(tasks[2].R.data(), B12, B22, m2);
    std::memcpy(tasks[3].L.data(), A22, m2 * sizeof(real_t)); mat_sub(tasks[3].R.data(), B21, B11, m2);
    mat_add(tasks[4].L.data(), A11, A12, m2); std::memcpy(tasks[4].R.data(), B22, m2 * sizeof(real_t));
    mat_sub(tasks[5].L.data(), A21, A11, m2); mat_add(tasks[5].R.data(), B11, B12, m2);
    mat_sub(tasks[6].L.data(), A12, A22, m2); mat_add(tasks[6].R.data(), B21, B22, m2);

    return tasks;
}

void combine_M(real_t* C, std::size_t n, std::vector<Task>& tasks) {
    const std::size_t m = n / 2, m2 = m * m;
    const real_t* M1 = tasks[0].M.data(); const real_t* M2 = tasks[1].M.data();
    const real_t* M3 = tasks[2].M.data(); const real_t* M4 = tasks[3].M.data();
    const real_t* M5 = tasks[4].M.data(); const real_t* M6 = tasks[5].M.data();
    const real_t* M7 = tasks[6].M.data();
    std::vector<real_t> C11(m2), C12(m2), C21(m2), C22(m2), tmp(m2);

    mat_add(tmp.data(),  M1, M4, m2);
    mat_sub(C11.data(), tmp.data(), M5, m2);
    mat_add(C11.data(), C11.data(), M7, m2);
    mat_add(C12.data(), M3, M5, m2);
    mat_add(C21.data(), M2, M4, m2);
    mat_sub(tmp.data(),  M1, M2, m2);
    mat_add(C22.data(), tmp.data(), M3, m2);
    mat_add(C22.data(), C22.data(), M6, m2);

    join4(C, n, C11.data(), C12.data(), C21.data(), C22.data());
}

// ---------------------------------------------------------------------------
// Kernel chinh: lai ghep MPI + OpenMP.
// Giong strassen_mpi nhung worker goi strassen_omp() thay vi strassen_seq().
// ---------------------------------------------------------------------------
void strassen_hybrid_run(
    real_t*       C,
    const real_t* A,
    const real_t* B,
    std::size_t   n,
    int           cutoff,
    int           depth,   // OpenMP task depth
    int           rank,
    int           world_size,
    double*       t_compute_out,
    double*       t_comm_out)
{
    // Xu ly n le: pad
    if (n % 2 != 0) {
        const std::size_t np = n + 1;
        real_t* Ap = nullptr;
        real_t* Bp = nullptr;
        real_t* Cp = nullptr;
        if (rank == 0) {
            Ap = alloc_buf(np * np);
            Bp = alloc_buf(np * np);
            Cp = alloc_buf(np * np);
            std::fill(Ap, Ap + np * np, 0.0);
            std::fill(Bp, Bp + np * np, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                std::memcpy(Ap + i * np, A + i * n, n * sizeof(real_t));
                std::memcpy(Bp + i * np, B + i * n, n * sizeof(real_t));
            }
        }
        strassen_hybrid_run(Cp, Ap, Bp, np, cutoff, depth,
                            rank, world_size, t_compute_out, t_comm_out);
        if (rank == 0) {
            for (std::size_t i = 0; i < n; ++i)
                std::memcpy(C + i * n, Cp + i * np, n * sizeof(real_t));
            std::free(Ap); std::free(Bp); std::free(Cp);
        }
        return;
    }

    const std::size_t m  = n / 2;
    const std::size_t m2 = m * m;
    const int task_elems = static_cast<int>(2 * m2);
    const int res_elems  = static_cast<int>(m2);

    std::vector<int> task_owners(7);
    for (int i = 0; i < 7; ++i) task_owners[i] = i % world_size;

    std::vector<Task> tasks;
    double comm_local = 0.0;

    // ---- Rank 0: phan ra, gui toan hang ----------------------------------
    if (rank == 0) {
        std::vector<real_t> A11(m2), A12(m2), A21(m2), A22(m2);
        std::vector<real_t> B11(m2), B12(m2), B21(m2), B22(m2);
        split4(A, n, A11.data(), A12.data(), A21.data(), A22.data());
        split4(B, n, B11.data(), B12.data(), B21.data(), B22.data());
        tasks = build_tasks(
            A11.data(), A12.data(), A21.data(), A22.data(),
            B11.data(), B12.data(), B21.data(), B22.data(), m);

        const double t0 = MPI_Wtime();
        std::vector<real_t> send_buf(2 * m2);
        for (int i = 0; i < 7; ++i) {
            int owner = task_owners[i];
            if (owner == 0) continue;
            std::memcpy(send_buf.data(),      tasks[i].L.data(), m2 * sizeof(real_t));
            std::memcpy(send_buf.data() + m2, tasks[i].R.data(), m2 * sizeof(real_t));
            MPI_Send(send_buf.data(), task_elems, MPI_DOUBLE, owner, i, MPI_COMM_WORLD);
        }
        comm_local += MPI_Wtime() - t0;
    } else {
        const double t0 = MPI_Wtime();
        std::vector<real_t> recv_buf(2 * m2);
        for (int i = 0; i < 7; ++i) {
            if (task_owners[i] != rank) continue;
            MPI_Recv(recv_buf.data(), task_elems, MPI_DOUBLE,
                     0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Task t;
            t.L.assign(recv_buf.data(),      recv_buf.data() + m2);
            t.R.assign(recv_buf.data() + m2, recv_buf.data() + 2 * m2);
            t.M.resize(m2, 0.0);
            tasks.push_back(std::move(t));
        }
        comm_local += MPI_Wtime() - t0;
    }

    // ---- Moi rank tinh bang strassen_omp (da luong) ----------------------
    // MPI_THREAD_FUNNELED: chi luong chinh goi MPI, cac luong OMP chi tinh.
    const double compute_start = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < 7; ++i) {
            if (task_owners[i] != 0) continue;
            // strassen_omp: mo vung parallel ben trong
            strassen_omp(tasks[i].M.data(),
                         tasks[i].L.data(),
                         tasks[i].R.data(),
                         m, cutoff, depth);
        }
    } else {
        int local_idx = 0;
        for (int i = 0; i < 7; ++i) {
            if (task_owners[i] != rank) continue;
            strassen_omp(tasks[local_idx].M.data(),
                         tasks[local_idx].L.data(),
                         tasks[local_idx].R.data(),
                         m, cutoff, depth);
            ++local_idx;
        }
    }

    const double compute_local = MPI_Wtime() - compute_start;

    // ---- Thu gom M_i -------------------------------------------------------
    if (rank == 0) {
        const double t0 = MPI_Wtime();
        for (int i = 0; i < 7; ++i) {
            int owner = task_owners[i];
            if (owner == 0) continue;
            MPI_Recv(tasks[i].M.data(), res_elems, MPI_DOUBLE,
                     owner, 100 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        comm_local += MPI_Wtime() - t0;
    } else {
        const double t0 = MPI_Wtime();
        int local_idx = 0;
        for (int i = 0; i < 7; ++i) {
            if (task_owners[i] != rank) continue;
            MPI_Send(tasks[local_idx].M.data(), res_elems, MPI_DOUBLE,
                     0, 100 + i, MPI_COMM_WORLD);
            ++local_idx;
        }
        comm_local += MPI_Wtime() - t0;
    }

    if (rank == 0) {
        combine_M(C, n, tasks);
    }

    double max_compute = 0.0, max_comm = 0.0;
    MPI_Reduce(&compute_local, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_local,    &max_comm,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        *t_compute_out = max_compute;
        *t_comm_out    = max_comm;
    }
}

} // namespace

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_FUNNELED && rank == 0)
        std::cerr << "Warning: MPI does not fully support MPI_THREAD_FUNNELED.\n";

    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "Argument error: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const std::size_t n = args.n;

    if (n == 0 || n > static_cast<std::size_t>(INT_MAX) ||
        n * n > static_cast<std::size_t>(INT_MAX)) {
        if (rank == 0) std::cerr << "Matrix size invalid or too large.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Thiet lap so thread OpenMP cho moi rank
    omp_set_num_threads(args.T);
    omp_set_nested(0); // Tranh nested parallelism khi P*T > so loi

    // ---- P == 1: dung strassen_omp truc tiep (khong MPI overhead) ---------
    if (world_size == 1) {
        real_t* A = alloc_matrix(n);
        real_t* B = alloc_matrix(n);
        real_t* C = alloc_matrix(n);

        gen_matrix(A, n, args.seed,     0, n);
        gen_matrix(B, n, args.seed + 1, 0, n);

        for (int w = 0; w < args.warmup; ++w)
            strassen_omp(C, A, B, n, args.cutoff, args.depth);

        int exit_code = 0;
        for (int run = 1; run <= args.runs; ++run) {
            const double t0 = MPI_Wtime();
            strassen_omp(C, A, B, n, args.cutoff, args.depth);
            const double elapsed = MPI_Wtime() - t0;

            bool verified = false;
            double rel_err = 0.0;
            if (args.verify) {
                verified = verify_freivalds(A, B, C, n, 1e-8, &rel_err);
                if (n <= 1000) {
                    real_t* Cref = alloc_matrix(n);
                    // Dung matmul_ref (chi co trong namespace, ko goi tu ngoai)
                    // => dung verify_full voi ket qua strassen_seq
                    strassen_seq(Cref, A, B, n, args.cutoff);
                    double fe = 0.0;
                    const bool fok = verify_full(C, Cref, n, 1e-8, &fe);
                    rel_err = std::max(rel_err, fe);
                    verified = verified && fok;
                    std::free(Cref);
                }
                if (!verified) exit_code = 1;
            }

            RunInfo info;
            info.algo      = "strassen";
            info.model     = "hybrid";
            info.n         = n;
            info.P         = 1;
            info.T         = args.T;
            info.run       = run;
            info.t_total   = elapsed;
            info.t_compute = elapsed;
            info.t_comm    = 0.0;
            info.verified  = verified;
            info.rel_err   = rel_err;
            info.cutoff    = args.cutoff;
            print_csv(info);
        }

        std::free(A); std::free(B); std::free(C);
        MPI_Finalize();
        return exit_code;
    }

    // ---- P > 1 -----------------------------------------------------------
    real_t* A = nullptr;
    real_t* B = nullptr;
    real_t* C = nullptr;

    if (rank == 0) {
        A = alloc_matrix(n);
        B = alloc_matrix(n);
        C = alloc_matrix(n);
        if (!A || !B || !C) {
            std::cerr << "Rank 0: allocation failed.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        gen_matrix(A, n, args.seed,     0, n);
        gen_matrix(B, n, args.seed + 1, 0, n);
    }

    // ---- Warm-up ----------------------------------------------------------
    for (int w = 0; w < args.warmup; ++w) {
        double tc = 0.0, tmm = 0.0;
        strassen_hybrid_run(C, A, B, n, args.cutoff, args.depth,
                            rank, world_size, &tc, &tmm);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ---- Cac luot do chinh thuc -------------------------------------------
    int exit_code = 0;

    for (int run = 1; run <= args.runs; ++run) {
        double t_compute = 0.0, t_comm = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        const double total_start = MPI_Wtime();

        strassen_hybrid_run(C, A, B, n, args.cutoff, args.depth,
                            rank, world_size, &t_compute, &t_comm);

        const double total_local = MPI_Wtime() - total_start;
        double t_total = 0.0;
        MPI_Reduce(&total_local, &t_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        bool verified = false;
        double rel_err = 0.0;
        if (rank == 0 && args.verify) {
            verified = verify_freivalds(A, B, C, n, 1e-8, &rel_err);
            if (n <= 1000) {
                real_t* Cref = alloc_matrix(n);
                strassen_seq(Cref, A, B, n, args.cutoff);
                double fe = 0.0;
                const bool fok = verify_full(C, Cref, n, 1e-8, &fe);
                rel_err = std::max(rel_err, fe);
                verified = verified && fok;
                std::free(Cref);
            }
            if (!verified) {
                std::cerr << "Verification FAILED: n=" << n
                          << ", P=" << world_size
                          << ", T=" << args.T
                          << ", run=" << run << '\n';
                exit_code = 1;
            }
        }

        if (rank == 0) {
            RunInfo info;
            info.algo      = "strassen";
            info.model     = "hybrid";
            info.n         = n;
            info.P         = world_size;
            info.T         = args.T;
            info.run       = run;
            info.t_total   = t_total;
            info.t_compute = t_compute;
            info.t_comm    = t_comm;
            info.verified  = args.verify ? verified : false;
            info.rel_err   = rel_err;
            info.cutoff    = args.cutoff;
            print_csv(info);
        }

        int global_err = 0;
        MPI_Allreduce(&exit_code, &global_err, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        exit_code = global_err;
    }

    std::free(A); std::free(B); std::free(C);
    MPI_Finalize();
    return exit_code;
}
