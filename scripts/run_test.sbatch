#!/bin/bash
#SBATCH --job-name=strassen_test
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Di chuyen vao dung thu muc
cd $HOME/Matrix_multiplication || exit 1
module load openmpi/5.0

# 1. Tao thu muc chua file thuc thi
mkdir -p bin results/cluster/logs

# 2. Tu dong bien dich neu chua co binary
echo "=== DANG BIEN DICH MÃ NGUỒN ==="
mpicxx -std=c++17 -O3 -march=native -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_mpi.cpp -o bin/strassen_mpi

mpicxx -std=c++17 -O3 -march=native -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_hybrid.cpp -o bin/strassen_hybrid

# 3. Chay thuc thi bang duong dan tuyet doi
echo "=== 1. CHAY THU STRASSEN MPI ==="
mpirun $HOME/Matrix_multiplication/bin/strassen_mpi -n 500 --verify

echo "=== 2. CHAY THU STRASSEN HYBRID ==="
mpirun $HOME/Matrix_multiplication/bin/strassen_hybrid -n 500 -t 2 --verify

echo "=== HOAN THANH LUC: $(date) ==="
