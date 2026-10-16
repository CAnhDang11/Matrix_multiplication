#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <new>
#include <unistd.h>

// ------------------------------------------------------------
// Internal utilities
// ------------------------------------------------------------

namespace {

constexpr std::size_t ALIGNMENT = 64; // 64-byte alignment for cache lines and SIMD.
constexpr std::uint64_t GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double random_unit(std::uint64_t seed, std::uint64_t index) {
    const std::uint64_t bits =
        splitmix64(seed ^ (index * GOLDEN_RATIO));

    // Use the upper 53 bits to produce a double in [0, 1).
    return static_cast<double>(bits >> 11) *
           (1.0 / 9007199254740992.0);
}

std::size_t checked_num_elements(std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension n must be > 0");
    }

    if (n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error("Matrix element count overflow");
    }

    const std::size_t count = n * n;

    if (count > std::numeric_limits<std::size_t>::max() /
                    sizeof(real_t)) {
        throw std::overflow_error("Matrix byte size overflow");
    }

    return count;
}

std::string require_value(
    int& i,
    int argc,
    char** argv,
    const std::string& option
) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(
            "Missing value after " + option
        );
    }

    return argv[++i];
}

std::size_t parse_size(
    const std::string& value,
    const std::string& name
) {
    std::size_t consumed = 0;
    unsigned long long result = 0;

    try {
        result = std::stoull(value, &consumed);
    } catch (...) {
        throw std::invalid_argument("Invalid value for " + name);
    }

    if (consumed != value.size() || value.empty() ||
        value[0] == '-') {
        throw std::invalid_argument("Invalid value for " + name);
    }

    if (result == 0 ||
        result > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            name + " must be greater than zero"
        );
    }

    return static_cast<std::size_t>(result);
}

int parse_int(
    const std::string& value,
    const std::string& name
) {
    std::size_t consumed = 0;
    long long result = 0;

    try {
        result = std::stoll(value, &consumed);
    } catch (...) {
        throw std::invalid_argument("Invalid value for " + name);
    }

    if (consumed != value.size() ||
        result < std::numeric_limits<int>::min() ||
        result > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid value for " + name);
    }

    return static_cast<int>(result);
}

void validate_args(const Args& args) {
    checked_num_elements(args.n);

    if (args.T <= 0) {
        throw std::invalid_argument("-t must be greater than zero");
    }

    if (args.runs <= 0) {
        throw std::invalid_argument("-r must be greater than zero");
    }

    if (args.cutoff <= 0) {
        throw std::invalid_argument(
            "--cutoff must be greater than zero"
        );
    }

    if (args.depth < 0) {
        throw std::invalid_argument("--depth must be >= 0");
    }

    if (args.warmup < 0) {
        throw std::invalid_argument("--warmup must be >= 0");
    }
}

} // namespace

// ------------------------------------------------------------
// Aligned memory allocation
// ------------------------------------------------------------

real_t* alloc_matrix(std::size_t n) {
    const std::size_t count = checked_num_elements(n);
    const std::size_t bytes = count * sizeof(real_t);

    void* ptr = nullptr;

    // POSIX allocation is available on macOS and Linux.
    const int rc = posix_memalign(&ptr, ALIGNMENT, bytes);

    if (rc != 0 || ptr == nullptr) {
        throw std::bad_alloc();
    }

    return static_cast<real_t*>(ptr);
}

// ------------------------------------------------------------
// Deterministic matrix generation
// ------------------------------------------------------------

void gen_matrix(
    real_t* M,// Pointer to the matrix to be generated.
    std::size_t n, // Size of the matrix (n x n).
    std::uint64_t seed, // Seed for deterministic matrix generation.
    std::size_t row0, // Starting row index.
    std::size_t nrows // Number of rows to generate.
) {
    if (M == nullptr) {
        throw std::invalid_argument("gen_matrix: M is null");
    }

    if (n == 0 || row0 > n || nrows > n - row0) {
        throw std::invalid_argument(
            "gen_matrix: invalid row range"
        );
    }

    checked_num_elements(n);

    for (std::size_t i = 0; i < nrows; ++i) {
        const std::size_t global_row = row0 + i;

        for (std::size_t j = 0; j < n; ++j) {
            const std::uint64_t global_index =
                static_cast<std::uint64_t>(global_row) *
                    static_cast<std::uint64_t>(n) +
                static_cast<std::uint64_t>(j);

            M[i * n + j] = random_unit(seed, global_index);
        }
    }
}

// ------------------------------------------------------------
// Wall-clock timer
// ------------------------------------------------------------

double now() {
    return omp_get_wtime();
}

// ------------------------------------------------------------
// Freivalds verification
//
// Checks A * (B * r) approximately equals C * r.
// Uses three deterministic random 0/1 vectors.
// Complexity: O(n^2) time and O(n) auxiliary memory.
// ------------------------------------------------------------

bool verify_freivalds(
    const real_t* A,
    const real_t* B,
    const real_t* C,
    std::size_t n,
    double tol,
    double* rel_err
) {
    if (A == nullptr || B == nullptr || C == nullptr ||
        n == 0 || tol < 0.0) {
        throw std::invalid_argument(
            "verify_freivalds: invalid input"
        );
    }

    checked_num_elements(n);

    constexpr int TRIALS = 3;

    std::vector<real_t> r(n);
    std::vector<real_t> br(n, 0.0);
    std::vector<real_t> abr(n, 0.0);
    std::vector<real_t> cr(n, 0.0);

    double worst_rel_err = 0.0;
    bool passed = true;

    for (int trial = 0; trial < TRIALS; ++trial) {
        const std::uint64_t trial_seed =
            0xd1b54a32d192ed03ULL +
            static_cast<std::uint64_t>(trial);

        for (std::size_t j = 0; j < n; ++j) {
            r[j] = static_cast<real_t>(
                splitmix64(trial_seed + j) & 1ULL
            );
        }

        std::fill(br.begin(), br.end(), 0.0);
        std::fill(abr.begin(), abr.end(), 0.0);
        std::fill(cr.begin(), cr.end(), 0.0);

        // br = B * r
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;

            for (std::size_t j = 0; j < n; ++j) {
                sum += B[i * n + j] * r[j];
            }

            br[i] = sum;
        }

        // abr = A * br
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;

            for (std::size_t j = 0; j < n; ++j) {
                sum += A[i * n + j] * br[j];
            }

            abr[i] = sum;
        }

        // cr = C * r
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;

            for (std::size_t j = 0; j < n; ++j) {
                sum += C[i * n + j] * r[j];
            }

            cr[i] = sum;
        }

        double error_sq = 0.0;
        double reference_sq = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            const double diff = abr[i] - cr[i];

            error_sq += diff * diff;
            reference_sq += abr[i] * abr[i];
        }

        const double denominator =
            std::max(std::sqrt(reference_sq), 1e-30);

        const double trial_rel_err =
            std::sqrt(error_sq) / denominator;

        worst_rel_err = std::max(worst_rel_err, trial_rel_err);

        if (!std::isfinite(trial_rel_err) ||
            trial_rel_err > tol) {
            passed = false;
        }
    }

    if (rel_err != nullptr) {
        *rel_err = worst_rel_err;
    }

    return passed;
}

// ------------------------------------------------------------
// Full element-by-element verification
// ------------------------------------------------------------

bool verify_full(
    const real_t* C,
    const real_t* Cref,
    std::size_t n,
    double tol,
    double* rel_err
) {
    if (C == nullptr || Cref == nullptr ||
        n == 0 || tol < 0.0) {
        throw std::invalid_argument(
            "verify_full: invalid input"
        );
    }

    const std::size_t count = checked_num_elements(n);

    double error_sq = 0.0;
    double reference_sq = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const double diff = C[i] - Cref[i];

        error_sq += diff * diff;
        reference_sq += Cref[i] * Cref[i];
    }

    const double denominator =
        std::max(std::sqrt(reference_sq), 1e-30);

    const double error = std::sqrt(error_sq) / denominator;

    if (rel_err != nullptr) {
        *rel_err = error;
    }

    return std::isfinite(error) && error <= tol;
}

// ------------------------------------------------------------
// Command-line argument parsing
// ------------------------------------------------------------

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];

        if (option == "-n") {
            args.n = parse_size(
                require_value(i, argc, argv, option), option
            );
        } else if (option == "-t") {
            args.T = parse_int(
                require_value(i, argc, argv, option), option
            );
        } else if (option == "-r") {
            args.runs = parse_int(
                require_value(i, argc, argv, option), option
            );
        } else if (option == "-s") {
            const std::string value =
                require_value(i, argc, argv, option);

            std::size_t consumed = 0;

            try {
                args.seed = std::stoull(value, &consumed);
            } catch (...) {
                throw std::invalid_argument("Invalid seed");
            }

            if (consumed != value.size() ||
                value.empty() || value[0] == '-') {
                throw std::invalid_argument("Invalid seed");
            }
        } else if (option == "--verify") {
            args.verify = true;
        } else if (option == "--cutoff") {
            args.cutoff = parse_int(
                require_value(i, argc, argv, option), option
            );
        } else if (option == "--depth") {
            args.depth = parse_int(
                require_value(i, argc, argv, option), option
            );
        } else if (option == "--kernel") {
            const std::string value =
                require_value(i, argc, argv, option);

            if (value == "ikj") {
                args.kernel = 0;
            } else if (value == "ijk") {
                args.kernel = 1;
            } else {
                throw std::invalid_argument(
                    "--kernel must be ijk or ikj"
                );
            }
        } else if (option == "--warmup") {
            const std::string value =
                require_value(i, argc, argv, option);

            // Warmup can be zero.
            std::size_t consumed = 0;
            long long result = 0;

            try {
                result = std::stoll(value, &consumed);
            } catch (...) {
                throw std::invalid_argument("Invalid --warmup");
            }

            if (consumed != value.size() || result < 0 ||
                result > std::numeric_limits<int>::max()) {
                throw std::invalid_argument(
                    "--warmup must be >= 0"
                );
            }

            args.warmup = static_cast<int>(result);
        } else if (option == "-h" || option == "--help") {
            std::cout
                << "Usage: program [options]\n"
                << "  -n N             Matrix dimension (default 1000)\n"
                << "  -t T             Thread count (default 1)\n"
                << "  -r R             Measured runs (default 5)\n"
                << "  -s SEED          Random seed (default 42)\n"
                << "  --verify         Enable correctness verification\n"
                << "  --cutoff N       Strassen cutoff (default 128)\n"
                << "  --depth N        Strassen parallel depth (default 2)\n"
                << "  --kernel ijk|ikj Select naive kernel (default ikj)\n"
                << "  --warmup N       Warmup runs (default 1)\n";

            std::exit(0);
        } else {
            throw std::invalid_argument(
                "Unknown option: " + option
            );
        }
    }

    validate_args(args);
    return args;
}

// ------------------------------------------------------------
// CSV output
// ------------------------------------------------------------

void print_csv(const RunInfo& r) {

    static bool header_printed = false;

    if (!header_printed) {
        std::cout
            << "algo,model,n,P,T,run,t_total,t_compute,t_comm,"
            << "verified,rel_err,cutoff,host,commit\n";
        header_printed = true;
    }

    char hostname[256] = "unknown";

    if (gethostname(hostname, sizeof(hostname) - 1) != 0) {
        std::strcpy(hostname, "unknown");
    }

    const char* commit = std::getenv("GIT_COMMIT");

    if (commit == nullptr || commit[0] == '\0') {
        commit = "unknown";
    }

    std::cout
        << std::setprecision(10)
        << (r.algo ? r.algo : "unknown") << ','
        << (r.model ? r.model : "unknown") << ','
        << r.n << ','
        << r.P << ','
        << r.T << ','
        << r.run << ','
        << r.t_total << ','
        << r.t_compute << ','
        << r.t_comm << ','
        << (r.verified ? "true" : "false") << ','
        << r.rel_err << ','
        << r.cutoff << ','
        << hostname << ','
        << commit << '\n';
}