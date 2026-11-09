#!/bin/bash
#SBATCH --job-name=strassen_bench
#SBATCH --output=results/cluster/bench_strassen_%j.csv
#SBATCH --error=results/cluster/bench_strassen_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00

# Di chuyen vao dung thu muc chua ma nguon
cd $HOME/Matrix_multiplication || exit 1
module load openmpi/5.0

mkdir -p bin results/cluster

echo "# =========================================================="
echo "# BAT DAU BENCHMARK STRASSEN CLUSTER (TV3 · D7)"
echo "# Host: $(hostname) | Thoi gian: $(date)"
echo "# =========================================================="

# 1. Benchmark Strassen MPI: Quet qua P in {1, 2, 4} va cac kich thuoc n
for n in 500 1000 2000 4000 8000 10000; do
    for P in 1 2 4; do
        mpirun --oversubscribe -np $P ./bin/strassen_mpi -n $n -r 3 --cutoff 128 --verify
    done
done

# 2. Benchmark Strassen Hybrid: Quet qua P in {2, 4} voi T=2 va cac kich thuoc n
for n in 500 1000 2000 4000 8000 10000; do
    for P in 2 4; do
        mpirun --oversubscribe -np $P ./bin/strassen_hybrid -n $n -t 2 -r 3 --cutoff 128 --verify
    done
done

echo "# =========================================================="
echo "# HOAN THANH BENCHMARK LUC: $(date)"
echo "# =========================================================="

