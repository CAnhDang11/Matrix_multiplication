#!/usr/bin/env bash
set -uo pipefail

echo "===== System Information ====="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"

if [[ "$(uname -s)" == "Darwin" ]]; then
    sw_vers
    echo
    echo "CPU model:"
    sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model
    echo "Logical CPUs: $(sysctl -n hw.logicalcpu)"
    echo "Physical CPUs: $(sysctl -n hw.physicalcpu)"
    echo "Memory (bytes): $(sysctl -n hw.memsize)"
else
    uname -a
    echo "Logical CPUs: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
    echo "CPU model:"
    grep -m1 -E 'model name|Hardware|Processor' /proc/cpuinfo 2>/dev/null || true
    echo "Memory:"
    grep -E 'MemTotal|MemAvailable' /proc/meminfo 2>/dev/null || true
fi

echo
echo "===== Compilers ====="
for compiler in g++-16 g++ clang++ mpicxx mpirun; do
    if command -v "$compiler" >/dev/null 2>&1; then
        echo "--- $compiler"
        command -v "$compiler"
        "$compiler" --version 2>&1 | head -n 2 || true
    else
        echo "--- $compiler: not found"
    fi
done

echo
echo "===== OpenMP Check ====="
TMPDIR_OMP="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_OMP"' EXIT

cat > "$TMPDIR_OMP/check.cpp" <<'CPP'
#include <iostream>
#include <omp.h>

int main() {
    int count = 0;

    #pragma omp parallel reduction(+:count)
    count += 1;

    std::cout << "OpenMP threads: " << count << '\n';
}
CPP

if command -v g++-16 >/dev/null 2>&1 &&
   g++-16 -std=c++17 -fopenmp "$TMPDIR_OMP/check.cpp" \
       -o "$TMPDIR_OMP/check"; then
    OMP_NUM_THREADS=4 "$TMPDIR_OMP/check"
else
    echo "OpenMP check skipped or failed."
fi

echo
echo "===== Environment Variables ====="
echo "CC=${CC:-unset}"
echo "CXX=${CXX:-unset}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset}"

# chmod +x scripts/sysinfo.sh