#!/usr/bin/env bash
# tests/test_strassen_mpi.sh — TV3: Kiem thu tinh dung dan cua strassen_mpi
# va strassen_hybrid voi P in {1,2,3,4,7,8} va n in {1,2,3,7,16,100,257,1000}.
#
# Chay: ./tests/test_strassen_mpi.sh
# Yeu cau: bin/strassen_mpi va bin/strassen_hybrid phai ton tai.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

MPIRUN="${MPIRUN:-mpirun}"
CUTOFF="${CUTOFF:-32}"
PASS=0
FAIL=0

run_test() {
    local name="$1"; shift
    local tmplog
    tmplog="$(mktemp /tmp/mpi_strassen_XXXXXX.log)"
    if "$@" > "$tmplog" 2>&1; then
        echo "[PASS] $name"
        PASS=$(( PASS + 1 ))
    else
        echo "[FAIL] $name"
        cat "$tmplog"
        FAIL=$(( FAIL + 1 ))
    fi
    rm -f "$tmplog"
}

# Kiem tra binary
for bin in strassen_mpi strassen_hybrid; do
    if [[ ! -x "bin/$bin" ]]; then
        echo "Chua co bin/$bin — build truoc: mpicxx ... -o bin/$bin"
        exit 1
    fi
done

echo "===== Kiem thu strassen_mpi ====="

# P = 1 den 4 tren cac kich thuoc tieu bieu (bao gom so le, luy thua 2, lon)
for n in 1 2 3 7 16 100 257 1000; do
    for P in 1 2 3 4; do
        run_test "strassen_mpi_n${n}_P${P}" \
            $MPIRUN -np "$P" ./bin/strassen_mpi \
            -n "$n" -r 1 --cutoff "$CUTOFF" --verify
    done
done

# P = 7 va P = 8 (cac truong hop phu hop theo plan §3.8)
for n in 16 100 257; do
    for P in 7 8; do
        run_test "strassen_mpi_n${n}_P${P}" \
            $MPIRUN -np "$P" ./bin/strassen_mpi \
            -n "$n" -r 1 --cutoff "$CUTOFF" --verify
    done
done

echo ""
echo "===== Kiem thu strassen_hybrid ====="

# P x T voi cac kich thuoc tieu bieu
for n in 1 2 3 7 16 100 257; do
    for P in 1 2; do
        for T in 1 2; do
            run_test "strassen_hybrid_n${n}_P${P}_T${T}" \
                $MPIRUN -np "$P" ./bin/strassen_hybrid \
                -n "$n" -t "$T" -r 1 \
                --cutoff "$CUTOFF" --depth 2 --verify
        done
    done
done

# Kiem thu lon hon voi P=4, T=2
for n in 100 257 1000; do
    run_test "strassen_hybrid_n${n}_P4_T2" \
        $MPIRUN -np 4 ./bin/strassen_hybrid \
        -n "$n" -t 2 -r 1 \
        --cutoff "$CUTOFF" --depth 2 --verify
done

echo ""
echo "Strassen MPI/Hybrid: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
