#ifndef COMMON_H
#define COMMON_H

#include <cstddef>
#include <cstdint>

// All matrices use double precision, square row-major storage.
using real_t = double; 

using matmul_fn = void (*)(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n
); // Function pointer type for matrix multiplication kernels.

struct Args {
    std::size_t n = 1000; // Size of the matrix (n x n).
    int T = 1; // Number of OpenMP threads.
    int runs = 5; // Number of timed runs.
    std::uint64_t seed = 42; // Seed for deterministic matrix generation.
    bool verify = false; // Whether to verify the result.

    int cutoff = 128; // Strassen cutoff (if applicable).
    int depth = 2; // Strassen parallel depth (if applicable).

    // 0 = ikj, 1 = ijk
    int kernel = 0; // Naive kernel selection.

    int warmup = 1; // Number of warmup runs (not timed).
};

struct RunInfo {
    const char* algo = "naive";
    const char* model = "seq";

    std::size_t n = 0; // Size of the matrix. 
    int P = 1; // Number of MPI processes.
    int T = 1;  // Number of OpenMP threads.
    int run = 0; // Run number.

    double t_total = 0.0; // Total wall-clock time.
    double t_compute = 0.0; // Compute time.
    double t_comm = 0.0; // Communication time.

    bool verified = false; // Whether the result was verified.
    double rel_err = 0.0; // Relative error of the result.
    int cutoff = 0; // Strassen cutoff (if applicable).
};

// Memory and matrix generation.
real_t* alloc_matrix(std::size_t n); // Allocates an n x n matrix with 64-byte alignment.

void gen_matrix(
    real_t* M,
    std::size_t n,
    std::uint64_t seed,
    std::size_t row0,
    std::size_t nrows
); // Generates a matrix with the specified dimensions and seed.

// Wall-clock timer in seconds.
double now();

// Verification.
bool verify_freivalds(
    const real_t* A,
    const real_t* B,
    const real_t* C,
    std::size_t n,
    double tol,
    double* rel_err
); // Verifies that A * B approximately equals C using Freivalds' algorithm.

bool verify_full(
    const real_t* C,
    const real_t* Cref,
    std::size_t n,
    double tol,
    double* rel_err
); // Verifies the full matrix multiplication result.

// Command-line arguments.
Args parse_args(int argc, char** argv);

// CSV output.
void print_csv(const RunInfo& r);

// Naive matrix multiplication kernels.
// Implementations will be added in N3 and N4.
void matmul_ijk(
    real_t* C, const real_t* A, const real_t* B,
    std::size_t n
);

void matmul_ikj(
    real_t* C, const real_t* A, const real_t* B,
    std::size_t n
);

// A and C refer to local row blocks; B is the full matrix.
void matmul_ikj_rows(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t nrows,
    std::size_t n
);

// Strassen declarations for integration with TV2.
void strassen_seq(
    real_t* C, const real_t* A, const real_t* B,
    std::size_t n, int cutoff
);

void strassen_omp(
    real_t* C, const real_t* A, const real_t* B,
    std::size_t n, int cutoff, int depth
);

#endif // COMMON_H