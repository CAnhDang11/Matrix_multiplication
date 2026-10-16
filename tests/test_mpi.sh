#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MPIRUN="${MPIRUN:-mpirun}"
PASS=0
FAIL=0

if [[ ! -x bin/naive_mpi ]]; then
    echo "Build bin/naive_mpi first."
    exit 1
fi

run_test() {
    local name="$1"
    shift

    if "$@" > /tmp/matmul_mpi_test.log 2>&1; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        cat /tmp/matmul_mpi_test.log
        FAIL=$((FAIL + 1))
    fi
}

for np in 1 2 3 4; do
    for n in 1 2 3 7 16 100; do
        run_test "mpi_n${n}_p${np}" \
            "$MPIRUN" -np "$np" \
            ./bin/naive_mpi -n "$n" -r 1 --verify
    done
done

echo
echo "MPI: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]