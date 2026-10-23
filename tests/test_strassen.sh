#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift

    if "$@" > /tmp/matmul_strassen_test.log 2>&1; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        cat /tmp/matmul_strassen_test.log
        FAIL=$((FAIL + 1))
    fi
}

if [[ ! -x bin/strassen_seq || ! -x bin/strassen_omp ]]; then
    echo "Build bin/strassen_seq and bin/strassen_omp first."
    exit 1
fi

# Odd and non-power-of-two sizes must pad correctly.
for n in 1 2 3 7 16 100 257 1000; do
    run_test "strassen_seq_n${n}" \
        ./bin/strassen_seq -n "$n" -r 1 --cutoff 32 --verify
done

for n in 1 2 3 7 16 100 257 1000; do
    for t in 1 2 4; do
        run_test "strassen_omp_n${n}_t${t}" \
            ./bin/strassen_omp -n "$n" -t "$t" -r 1 \
            --cutoff 32 --depth 2 --verify
    done
done

echo
echo "Strassen: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
