#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift

    if "$@" > /tmp/matmul_test.log 2>&1; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        cat /tmp/matmul_test.log
        FAIL=$((FAIL + 1))
    fi
}

if [[ ! -x bin/naive_seq || ! -x bin/naive_omp ]]; then
    echo "Build bin/naive_seq and bin/naive_omp first."
    exit 1
fi

# Sequential: hai kernel ijk và ikj.
for n in 1 2 3 7 16 100 257; do
    run_test "seq_ijk_n${n}" \
        ./bin/naive_seq -n "$n" -r 1 --kernel ijk --verify

    run_test "seq_ikj_n${n}" \
        ./bin/naive_seq -n "$n" -r 1 --kernel ikj --verify
done

# OpenMP: nhiều số thread.
for n in 1 2 3 7 16 100 257; do
    for t in 1 2 4; do
        run_test "omp_ikj_n${n}_t${t}" \
            ./bin/naive_omp -n "$n" -t "$t" -r 1 \
            --kernel ikj --verify
    done
done

echo
echo "Sequential/OpenMP: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]