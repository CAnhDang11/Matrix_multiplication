#!/bin/bash
#SBATCH --job-name=strassen_bench
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00

# Di chuyen vao dung thu muc chua ma nguon
cd $HOME/Matrix_multiplication || exit 1
module load openmpi/5.0

mkdir -p bin results/cluster/logs

echo "=========================================================="
echo "BAT DAU BENCHMARK STRASSEN CLUSTER (TV3 · D7)"
echo "Host: $(hostname) | Ngay: $(date)"
echo "Danh sach cac node duoc cap phat: $SLURM_NODELIST"
echo "Tong so task MPI: $SLURM_NTASKS"
echo "=========================================================="

# Chay benchmark Strassen MPI & Strassen Hybrid voi danh sach n day du
# Quet qua cac so process MPI: 1, 2, 4, 8
./scripts/run_cluster.sh --only strassen_mpi,strassen_hybrid

echo "=========================================================="
echo "HOAN THANH BENCHMARK LUC: $(date)"
echo "Ket qua CSV da duoc luu tai thu muc: results/cluster/"
echo "=========================================================="
