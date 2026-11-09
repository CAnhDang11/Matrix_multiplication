#!/bin/bash
#SBATCH --job-name=strassen_test
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

module load openmpi/5.0

echo "=== 1. CHAY THU STRASSEN MPI ==="
mpirun -np 2 ./bin/strassen_mpi -n 500 --verify

echo "=== 2. CHAY THU STRASSEN HYBRID ==="
mpirun -np 2 ./bin/strassen_hybrid -n 500 -t 2 --verify

echo "=== HOAN THANH LUC: $(date) ==="
