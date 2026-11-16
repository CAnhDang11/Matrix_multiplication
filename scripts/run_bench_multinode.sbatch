#!/bin/bash
#SBATCH --job-name=strassen_multinode
#SBATCH --output=results/cluster/bench_strassen_multi_%j.out
#SBATCH --error=results/cluster/bench_strassen_multi_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00

# Di chuyen vao dung thu muc chua ma nguon
cd $HOME/Matrix_multiplication || exit 1
module load openmpi/5.0

mkdir -p bin results/cluster

# Bien dich lai dam bao binary moi nhat tren cluster
mpicxx -std=c++17 -O3 -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_mpi.cpp -o bin/strassen_mpi

mpicxx -std=c++17 -O3 -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_hybrid.cpp -o bin/strassen_hybrid

# Git commit hash cho cot "commit" trong CSV
GIT_COMMIT="$(git describe --tags --always 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
export GIT_COMMIT

CSV_FILE="results/cluster/bench_strassen_multi_${SLURM_JOB_ID}.csv"
HEADER="algo,model,n,P,T,run,t_total,t_compute,t_comm,verified,rel_err,cutoff,host,commit"
echo "$HEADER" > "$CSV_FILE"

echo "# =========================================================="
echo "# BAT DAU BENCHMARK MULTI-NODE STRASSEN CLUSTER (TV3 · D7)"
echo "# Nodes: $SLURM_NODELIST (Total: $SLURM_NNODES nodes)"
echo "# Tasks: $SLURM_NTASKS (ntasks-per-node=2)"
echo "# Thoi gian: $(date)"
echo "# Commit: $GIT_COMMIT"
echo "# =========================================================="

# 1. Benchmark Strassen MPI Multi-Node: P=2, 4
for n in 1000 2000 4000 8000 10000; do
    for P in 2 4; do
        mpirun -np $P ./bin/strassen_mpi -n $n -r 3 --cutoff 128 --verify >> "$CSV_FILE"
    done
done

# 2. Benchmark Strassen Hybrid Multi-Node: P=2, 4 voi T=2
for n in 1000 2000 4000 8000 10000; do
    for P in 2 4; do
        mpirun -np $P ./bin/strassen_hybrid -n $n -t 2 -r 3 --cutoff 128 --verify >> "$CSV_FILE"
    done
done

echo "# =========================================================="
echo "# HOAN THANH BENCHMARK MULTI-NODE LUC: $(date)"
echo "# CSV: $CSV_FILE"
echo "# =========================================================="
