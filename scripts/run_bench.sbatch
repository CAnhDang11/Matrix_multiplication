#!/bin/bash
#SBATCH --job-name=strassen_bench
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Di chuyen vao dung thu muc
cd $HOME/Matrix_multiplication || exit 1
module load openmpi/5.0

mkdir -p bin results/cluster/logs
chmod +x scripts/*.sh tests/*.sh 2>/dev/null || true
sed -i 's/\r$//' scripts/*.sh tests/*.sh 2>/dev/null || true

# 1. Bien dich tuong thich 100% voi kien truc CPU cua node duoc cap phat
echo "=== BIEN DICH TREN $(hostname) ==="
mpicxx -std=c++17 -O3 -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_mpi.cpp -o bin/strassen_mpi

mpicxx -std=c++17 -O3 -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_hybrid.cpp -o bin/strassen_hybrid

echo "=========================================================="
echo "BAT DAU BENCHMARK STRASSEN CLUSTER (TV3 · D7)"
echo "Host: $(hostname) | Ngay: $(date)"
echo "Node tinh toan: $SLURM_NODELIST"
echo "Tong so task MPI: $SLURM_NTASKS"
echo "=========================================================="

# 2. Chay benchmark Strassen MPI & Strassen Hybrid
./scripts/run_cluster.sh --only strassen_mpi,strassen_hybrid

echo "=========================================================="
echo "HOAN THANH BENCHMARK LUC: $(date)"
echo "Ket qua CSV da duoc luu tai thu muc: results/cluster/"
echo "=========================================================="
