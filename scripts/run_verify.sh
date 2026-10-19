#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CXX="${CXX:-g++-16}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O3 -fopenmp}"
LOG_DIR="${LOG_DIR:-results/shared}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/verify_${TIMESTAMP}.log"

# Matrices to verify, including odd and non-power-of-two dimensions.
SIZES=(1 2 3 7 16 100 257)

# Override with TEST_THREADS="1 2 4" if needed.
read -r -a THREADS <<< "${TEST_THREADS:-1 2 4}"

FAILED=0
PASSED=0
SKIPPED=0

# Redirect all script output into both terminal and log.
exec > >(tee "$LOG_FILE") 2>&1

echo "===== TV1 Verification ====="
echo "Date: $(date)"
echo "Root: $ROOT"
echo "Compiler: $CXX"
echo "Flags: $CXXFLAGS"
echo

# Optional build step: compile only binaries whose source and main()
# have been implemented. Set BUILD=0 to skip.
if [[ "${BUILD:-1}" == "1" ]]; then
    echo "===== Build ====="

    if [[ -f src/common/common.cpp &&
          -f src/naive/naive_seq.cpp ]]; then
        # Expected source layout; adapt if your team chooses other names.
        if $CXX $CXXFLAGS -Iinclude \
            src/common/common.cpp src/naive/naive_seq.cpp \
            -o bin/naive_seq; then
            echo "Built bin/naive_seq"
        else
            echo "BUILD FAILED: naive_seq"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "Skipping naive_seq build: expected source files not found."
        SKIPPED=$((SKIPPED + 1))
    fi

    if [[ -f src/common/common.cpp &&
          -f src/naive/naive_omp.cpp ]]; then
        if $CXX $CXXFLAGS -Iinclude \
            src/common/common.cpp src/naive/naive_omp.cpp \
            -o bin/naive_omp; then
            echo "Built bin/naive_omp"
        else
            echo "BUILD FAILED: naive_omp"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "Skipping naive_omp build: expected source files not found."
        SKIPPED=$((SKIPPED + 1))
    fi

    if [[ -f src/common/common.cpp &&
          -f src/strassen/strassen.cpp &&
          -f src/strassen/strassen_seq.cpp ]]; then
        if $CXX $CXXFLAGS -Iinclude \
            src/common/common.cpp \
            src/strassen/strassen.cpp \
            src/strassen/strassen_seq.cpp \
            -o bin/strassen_seq; then
            echo "Built bin/strassen_seq"
        else
            echo "BUILD FAILED: strassen_seq"
            FAILED=$((FAILED + 1))
        fi
    fi

    if [[ -f src/common/common.cpp &&
          -f src/strassen/strassen.cpp &&
          -f src/strassen/strassen_omp.cpp ]]; then
        if $CXX $CXXFLAGS -Iinclude \
            src/common/common.cpp \
            src/strassen/strassen.cpp \
            src/strassen/strassen_omp.cpp \
            -o bin/strassen_omp; then
            echo "Built bin/strassen_omp"
        else
            echo "BUILD FAILED: strassen_omp"
            FAILED=$((FAILED + 1))
        fi
    fi
fi

echo
echo "===== Correctness Tests ====="

# Each binary must accept: -n N -t T -r 1 --verify.
# Additional binaries can be supplied through VERIFY_BINS.
BINS=(bin/naive_seq bin/naive_omp bin/strassen_seq bin/strassen_omp)
if [[ -n "${VERIFY_BINS:-}" ]]; then
    read -r -a BINS <<< "$VERIFY_BINS"
fi

for bin in "${BINS[@]}"; do
    if [[ ! -x "$bin" ]]; then
        echo "SKIP: $bin (not built or not executable)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    for n in "${SIZES[@]}"; do
        for t in "${THREADS[@]}"; do
            echo
            echo "TEST: $bin -n $n -t $t"

            if "$bin" -n "$n" -t "$t" -r 1 -s 42 --verify; then
                echo "PASS: $bin n=$n t=$t"
                PASSED=$((PASSED + 1))
            else
                echo "FAIL: $bin n=$n t=$t"
                FAILED=$((FAILED + 1))
            fi
        done
    done
done

echo
echo "===== Summary ====="
echo "Passed:  $PASSED"
echo "Failed:  $FAILED"
echo "Skipped: $SKIPPED"
echo "Log: $LOG_FILE"

if (( FAILED > 0 )); then
    exit 1
fi

if (( PASSED == 0 )); then
    echo "ERROR: No tests were executed. Implement/build the binaries first."
    exit 2
fi

echo "Verification completed."


#chmod +x scripts/run_verify.sh