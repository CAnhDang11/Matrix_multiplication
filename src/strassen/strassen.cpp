#include "common.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

struct MatrixBuf {
    real_t* p = nullptr;

    explicit MatrixBuf(std::size_t n) : p(alloc_matrix(n)) {}

    ~MatrixBuf() {
        std::free(p);
    }

    MatrixBuf(const MatrixBuf&) = delete;
    MatrixBuf& operator=(const MatrixBuf&) = delete;

    real_t* get() const {
        return p;
    }
};

// Same access pattern as matmul_ikj (TV1 leaf kernel).
void leaf_ikj(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n
) {
    std::fill(C, C + n * n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            const real_t aik = A[i * n + k];
            const real_t* brow = B + k * n;
            real_t* crow = C + i * n;

            for (std::size_t j = 0; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

void add_mat(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n
) {
    const std::size_t count = n * n;
    for (std::size_t i = 0; i < count; ++i) {
        C[i] = A[i] + B[i];
    }
}

void sub_mat(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n
) {
    const std::size_t count = n * n;
    for (std::size_t i = 0; i < count; ++i) {
        C[i] = A[i] - B[i];
    }
}

void split_quadrants(
    real_t* A11,
    real_t* A12,
    real_t* A21,
    real_t* A22,
    const real_t* A,
    std::size_t n
) {
    const std::size_t m = n / 2;

    for (std::size_t i = 0; i < m; ++i) {
        const real_t* row0 = A + i * n;
        const real_t* row1 = A + (i + m) * n;

        std::copy(row0, row0 + m, A11 + i * m);
        std::copy(row0 + m, row0 + n, A12 + i * m);
        std::copy(row1, row1 + m, A21 + i * m);
        std::copy(row1 + m, row1 + n, A22 + i * m);
    }
}

void join_quadrants(
    real_t* C,
    const real_t* C11,
    const real_t* C12,
    const real_t* C21,
    const real_t* C22,
    std::size_t n
) {
    const std::size_t m = n / 2;

    for (std::size_t i = 0; i < m; ++i) {
        real_t* row0 = C + i * n;
        real_t* row1 = C + (i + m) * n;

        std::copy(C11 + i * m, C11 + (i + 1) * m, row0);
        std::copy(C12 + i * m, C12 + (i + 1) * m, row0 + m);
        std::copy(C21 + i * m, C21 + (i + 1) * m, row1);
        std::copy(C22 + i * m, C22 + (i + 1) * m, row1 + m);
    }
}

void copy_into_padded(
    real_t* Dst,
    const real_t* Src,
    std::size_t n,
    std::size_t np
) {
    std::fill(Dst, Dst + np * np, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        std::copy(Src + i * n, Src + (i + 1) * n, Dst + i * np);
    }
}

void copy_from_padded(
    real_t* Dst,
    const real_t* Src,
    std::size_t n,
    std::size_t np
) {
    for (std::size_t i = 0; i < n; ++i) {
        std::copy(Src + i * np, Src + i * np + n, Dst + i * n);
    }
}

void strassen_rec(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n,
    int cutoff,
    int max_depth,
    int cur_depth,
    bool parallel
);

void strassen_even(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n,
    int cutoff,
    int max_depth,
    int cur_depth,
    bool parallel
) {
    const std::size_t m = n / 2;

    MatrixBuf A11(m), A12(m), A21(m), A22(m);
    MatrixBuf B11(m), B12(m), B21(m), B22(m);
    MatrixBuf M1(m), M2(m), M3(m), M4(m), M5(m), M6(m), M7(m);
    MatrixBuf L1(m), R1(m), L2(m), R2(m), L3(m), R3(m);
    MatrixBuf L4(m), R4(m), L5(m), R5(m), L6(m), R6(m), L7(m), R7(m);
    MatrixBuf T1(m), T2(m);

    split_quadrants(A11.get(), A12.get(), A21.get(), A22.get(), A, n);
    split_quadrants(B11.get(), B12.get(), B21.get(), B22.get(), B, n);

    // Operand pairs for M1..M7. Each pair has its own buffer so
    // OpenMP tasks never share a scratch matrix.
    add_mat(L1.get(), A11.get(), A22.get(), m);
    add_mat(R1.get(), B11.get(), B22.get(), m);

    add_mat(L2.get(), A21.get(), A22.get(), m);
    std::copy(B11.get(), B11.get() + m * m, R2.get());

    std::copy(A11.get(), A11.get() + m * m, L3.get());
    sub_mat(R3.get(), B12.get(), B22.get(), m);

    std::copy(A22.get(), A22.get() + m * m, L4.get());
    sub_mat(R4.get(), B21.get(), B11.get(), m);

    add_mat(L5.get(), A11.get(), A12.get(), m);
    std::copy(B22.get(), B22.get() + m * m, R5.get());

    sub_mat(L6.get(), A21.get(), A11.get(), m);
    add_mat(R6.get(), B11.get(), B12.get(), m);

    sub_mat(L7.get(), A12.get(), A22.get(), m);
    add_mat(R7.get(), B21.get(), B22.get(), m);

    const bool use_tasks = parallel && cur_depth < max_depth;

    if (use_tasks) {
        real_t* outs[7] = {
            M1.get(), M2.get(), M3.get(), M4.get(),
            M5.get(), M6.get(), M7.get()
        };
        real_t* lefts[7] = {
            L1.get(), L2.get(), L3.get(), L4.get(),
            L5.get(), L6.get(), L7.get()
        };
        real_t* rights[7] = {
            R1.get(), R2.get(), R3.get(), R4.get(),
            R5.get(), R6.get(), R7.get()
        };

        for (int k = 0; k < 7; ++k) {
            real_t* out = outs[k];
            real_t* left = lefts[k];
            real_t* right = rights[k];
            #pragma omp task default(none) firstprivate(out, left, right, m, cutoff, max_depth, cur_depth, parallel)
            strassen_rec(
                out, left, right, m, cutoff, max_depth,
                cur_depth + 1, parallel
            );
        }
        #pragma omp taskwait
    } else {
        strassen_rec(M1.get(), L1.get(), R1.get(), m, cutoff, max_depth, cur_depth + 1, parallel);
        strassen_rec(M2.get(), L2.get(), R2.get(), m, cutoff, max_depth, cur_depth + 1, parallel);
        strassen_rec(M3.get(), L3.get(), R3.get(), m, cutoff, max_depth, cur_depth + 1, parallel);
        strassen_rec(M4.get(), L4.get(), R4.get(), m, cutoff, max_depth, cur_depth + 1, parallel);
        strassen_rec(M5.get(), L5.get(), R5.get(), m, cutoff, max_depth, cur_depth + 1, parallel);
        strassen_rec(M6.get(), L6.get(), R6.get(), m, cutoff, max_depth, cur_depth + 1, parallel);
        strassen_rec(M7.get(), L7.get(), R7.get(), m, cutoff, max_depth, cur_depth + 1, parallel);
    }

    // C11 = M1 + M4 - M5 + M7
    add_mat(T1.get(), M1.get(), M4.get(), m);
    sub_mat(T2.get(), T1.get(), M5.get(), m);
    add_mat(A11.get(), T2.get(), M7.get(), m);

    // C12 = M3 + M5
    add_mat(A12.get(), M3.get(), M5.get(), m);

    // C21 = M2 + M4
    add_mat(A21.get(), M2.get(), M4.get(), m);

    // C22 = M1 - M2 + M3 + M6
    sub_mat(T1.get(), M1.get(), M2.get(), m);
    add_mat(T2.get(), T1.get(), M3.get(), m);
    add_mat(A22.get(), T2.get(), M6.get(), m);   

    join_quadrants(C, A11.get(), A12.get(), A21.get(), A22.get(), n);
}

void strassen_rec(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n,
    int cutoff,
    int max_depth,
    int cur_depth,
    bool parallel
) {
    if (n <= static_cast<std::size_t>(cutoff)) {
        leaf_ikj(C, A, B, n);
        return;
    }

    if (n % 2 != 0) {
        const std::size_t np = n + 1;
        MatrixBuf Ap(np), Bp(np), Cp(np);
        copy_into_padded(Ap.get(), A, n, np);
        copy_into_padded(Bp.get(), B, n, np);
        strassen_rec(
            Cp.get(), Ap.get(), Bp.get(), np, cutoff,
            max_depth, cur_depth, parallel
        );
        copy_from_padded(C, Cp.get(), n, np);
        return;
    }

    strassen_even(
        C, A, B, n, cutoff, max_depth, cur_depth, parallel
    );
}

} // namespace

void strassen_seq(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n,
    int cutoff
) {
    if (!C || !A || !B || n == 0 || cutoff <= 0) {
        throw std::invalid_argument("strassen_seq: invalid input");
    }

    strassen_rec(C, A, B, n, cutoff, 0, 0, false);
}

void strassen_omp(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n,
    int cutoff,
    int depth
) {
    if (!C || !A || !B || n == 0 || cutoff <= 0 || depth < 0) {
        throw std::invalid_argument("strassen_omp: invalid input");
    }

    #pragma omp parallel
    {
        #pragma omp single
        strassen_rec(C, A, B, n, cutoff, depth, 0, true);
    }
}
