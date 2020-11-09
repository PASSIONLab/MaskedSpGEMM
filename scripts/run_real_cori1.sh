#!/bin/bash

#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH --mail-user=isratnisa@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


export OMP_PLACES=cores
# export OMP_PROC_BIND=close
export OMP_NESTED=False
export OMP_MAX_ACTIVE_LEVELS=2


path=/global/homes/i/inisa/spgemm/outerspgemm/bin
output=/global/homes/i/inisa/spgemm/outerspgemm/output.txt
matpath=/project/projectdirs/m1982/israt/matrices

for mat in ${matpath}/* ; do
	echo "Processing $mat" >> $output
	# OuterSpGEMM Algorithm
	numactl --membind=0 $path/OuterSpGEMM_hw text $mat $mat ~/product.txt $OMP_NUM_THREADS 1024 32 >> $output
	# HashSpGEMM Algorithm
	numactl --membind=0 $path/HashSpGEMM_hw text $mat $mat ~/product.txt $OMP_NUM_THREADS 1024 32 >> $output
	# HeapSpGEMM Algorithm
	numactl --membind=0 $path/HeapSpGEMM_hw text $mat $mat ~/product.txt $OMP_NUM_THREADS 1024 32 >> $output

done
