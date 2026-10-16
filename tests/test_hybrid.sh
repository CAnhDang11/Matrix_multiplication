#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MPIRUN="${MPIRUN:-mpirun}"
PASS=0
FAIL=0

if [[ ! -x bin/naive_hybrid ]]; then
    echo "Build bin/naive_hybrid first."
    exit 1
fi

run_test() {
    local name="$1"
    shift

    if "$@" > /tmp/matmul_hybrid_test.log 2>&1; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        cat /tmp/matmul_hybrid_test.log
        FAIL=$((FAIL + 1))
    fi
}

# Bao gồm n < P và n không chia hết cho P.
for np in 1 2 3 4; do
    for n in 1 2 3 7 16 100; do
        for t in 1 2 4; do
            run_test "hybrid_n${n}_p${np}_t${t}" \
                "$MPIRUN" -np "$np" \
                ./bin/naive_hybrid -n "$n" -t "$t" \
                -r 1 --verify
        done
    done
done

echo
echo "Hybrid: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]