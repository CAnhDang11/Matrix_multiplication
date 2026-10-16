#include "common.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const char* name) {
    if (condition) {
        ++passed;
        std::cout << "[PASS] " << name << '\n';
    } else {
        ++failed;
        std::cerr << "[FAIL] " << name << '\n';
    }
}

void reference_matmul(
    real_t* C,
    const real_t* A,
    const real_t* B,
    std::size_t n
) {
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

int main() {
    // Test 1: cấp phát ma trận.
    {
        real_t* M = alloc_matrix(7);
        check(M != nullptr, "alloc_matrix returns valid memory");
        std::free(M);
    }

    // Test 2: sinh ma trận cùng seed phải cho cùng kết quả.
    {
        constexpr std::size_t n = 7;
        real_t* A = alloc_matrix(n);
        real_t* B = alloc_matrix(n);

        if (A == nullptr || B == nullptr) {
            check(false, "deterministic matrix generation");
        } else {
            gen_matrix(A, n, 42, 0, n);
            gen_matrix(B, n, 42, 0, n);

            bool same = true;
            for (std::size_t i = 0; i < n * n; ++i) {
                if (A[i] != B[i]) {
                    same = false;
                    break;
                }
            }

            check(same, "same seed produces same matrix");
        }

        std::free(A);
        std::free(B);
    }

    // Test 3: seed khác phải tạo ma trận khác.
    {
        constexpr std::size_t n = 7;
        real_t* A = alloc_matrix(n);
        real_t* B = alloc_matrix(n);

        if (A == nullptr || B == nullptr) {
            check(false, "different seeds produce different matrices");
        } else {
            gen_matrix(A, n, 42, 0, n);
            gen_matrix(B, n, 43, 0, n);

            bool different = false;
            for (std::size_t i = 0; i < n * n; ++i) {
                if (A[i] != B[i]) {
                    different = true;
                    break;
                }
            }

            check(different, "different seeds produce different matrices");
        }

        std::free(A);
        std::free(B);
    }

    // Test 4: verify_full chấp nhận hai ma trận giống nhau.
    {
        constexpr std::size_t n = 3;
        const real_t C[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        double rel_err = 0.0;
        const bool ok = verify_full(C, C, n, 1e-10, &rel_err);

        check(ok, "verify_full accepts identical matrices");
        check(rel_err < 1e-10, "identical matrices have small relative error");
    }

    // Test 5: verify_full phải phát hiện ma trận sai.
    {
        constexpr std::size_t n = 2;
        const real_t expected[] = {1, 2, 3, 4};
        const real_t incorrect[] = {1, 2, 3, 5};

        double rel_err = 0.0;
        const bool ok = verify_full(
            incorrect, expected, n, 1e-10, &rel_err
        );

        check(!ok, "verify_full rejects incorrect result");
    }

    // Test 6: kiểm tra Freivalds trên kết quả nhân ma trận.
    {
        constexpr std::size_t n = 4;
        real_t* A = alloc_matrix(n);
        real_t* B = alloc_matrix(n);
        real_t* C = alloc_matrix(n);

        if (A == nullptr || B == nullptr || C == nullptr) {
            check(false, "Freivalds verification");
        } else {
            gen_matrix(A, n, 42, 0, n);
            gen_matrix(B, n, 43, 0, n);
            reference_matmul(C, A, B, n);

            double rel_err = 0.0;
            const bool ok = verify_freivalds(
                A, B, C, n, 1e-10, &rel_err
            );

            check(ok, "Freivalds accepts correct matrix product");

            // Làm sai một phần tử và kiểm tra bằng full verification.
            C[0] += 100.0;

            double full_err = 0.0;
            const bool bad = verify_full(
                C, C, n, 1e-10, &full_err
            );

            // Hai đối số ở trên giống nhau, vì vậy phép thử này chỉ
            // kiểm tra lại API chứ chưa thể phát hiện C bị thay đổi.
            // Phần test so sánh đúng/sai được thực hiện bên dưới.
            (void)bad;
        }

        std::free(A);
        std::free(B);
        std::free(C);
    }

    // Test 7: Freivalds phải phát hiện kết quả bị sửa.
    {
        constexpr std::size_t n = 4;
        real_t* A = alloc_matrix(n);
        real_t* B = alloc_matrix(n);
        real_t* C = alloc_matrix(n);
        real_t* Cref = alloc_matrix(n);

        if (!A || !B || !C || !Cref) {
            check(false, "corrupted product is rejected");
        } else {
            gen_matrix(A, n, 42, 0, n);
            gen_matrix(B, n, 43, 0, n);
            reference_matmul(Cref, A, B, n);

            for (std::size_t i = 0; i < n * n; ++i) {
                C[i] = Cref[i];
            }

            C[0] += 100.0;

            double rel_err = 0.0;
            const bool ok = verify_freivalds(
                A, B, C, n, 1e-10, &rel_err
            );

            check(!ok, "Freivalds rejects corrupted matrix product");

            double full_err = 0.0;
            const bool full_ok = verify_full(
                C, Cref, n, 1e-10, &full_err
            );

            check(!full_ok, "full verification rejects corrupted product");
        }

        std::free(A);
        std::free(B);
        std::free(C);
        std::free(Cref);
    }

    std::cout << "\nSummary: " << passed << " passed, "
              << failed << " failed\n";

    return failed == 0 ? 0 : 1;
}