#!/bin/bash

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NESTED=False
export OMP_MAX_ACTIVE_LEVELS=2

#------
# OuterSpGEMM Algorithm
echo 'Running OuterSpGEMM experiments'
#------

# run on ER scale 20
echo 'OuterSpGEMM on ER-20'
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 20 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 20 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 20 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 20 16 $OMP_NUM_THREADS 8192 16

# run on ER scale 19
echo 'OuterSpGEMM on ER-19'
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 19 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 19 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 19 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 19 16 $OMP_NUM_THREADS 4096 16

# run on ER scale 18
echo 'OuterSpGEMM on ER-18'
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 18 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 18 4 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 18 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM_hw gen er 18 16 $OMP_NUM_THREADS 2048 32

# run on r-mat scale 16
echo 'OuterSpGEMM on RMAT-16'
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 16 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 16 4 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 16 8 $OMP_NUM_THREADS 4096 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 16 16 $OMP_NUM_THREADS 16384 32

# run on r-mat scale 15
echo 'OuterSpGEMM on RMAT-15'
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 15 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 15 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 15 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 15 16 $OMP_NUM_THREADS 8192 32

# run on r-mat scale 14
echo 'OuterSpGEMM on RMAT-14'
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 14 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 14 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 14 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/OuterSpGEMM2_hw gen rmat 14 16 $OMP_NUM_THREADS 4096 32

# run on ER scale 20
echo 'OuterSpGEMM bandwidth on ER-20'
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 20 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 20 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 20 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 20 16 $OMP_NUM_THREADS 4096 16

# run on ER scale 19
echo 'OuterSpGEMM bandwidth on ER-19'
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 19 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 19 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 19 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 19 16 $OMP_NUM_THREADS 4096 16

# run on ER scale 18
echo 'OuterSpGEMM bandwidth on ER-18'
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 18 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 18 4 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 18 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM_hw gen er 18 16 $OMP_NUM_THREADS 2048 32

# run on r-mat scale 16
echo 'OuterSpGEMM bandwidth on RMAT-16'
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 16 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 16 4 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 16 8 $OMP_NUM_THREADS 4096 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 16 16 $OMP_NUM_THREADS 4096 32

# run on r-mat scale 15
echo 'OuterSpGEMM bandwidth on RMAT-15'
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 15 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 15 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 15 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 15 16 $OMP_NUM_THREADS 4096 32

# run on r-mat scale 14
echo 'OuterSpGEMM bandwidth on RMAT-14'
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 14 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 14 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 14 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/ProfileOuterSpGEMM2_hw gen rmat 14 16 $OMP_NUM_THREADS 4096 32

#------
# Heap Algorithm
#------

# run on ER scale 20
echo 'HeapSpGEMM on ER-20'
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 20 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 20 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 20 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 20 16 $OMP_NUM_THREADS 8192 16

# run on ER scale 19
echo 'HeapSpGEMM on ER-19'
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 19 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 19 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 19 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 19 16 $OMP_NUM_THREADS 4096 16

# run on ER scale 18
echo 'HeapSpGEMM on ER-18'
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 18 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 18 4 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 18 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen er 18 16 $OMP_NUM_THREADS 2048 32

# run on r-mat scale 16
echo 'HeapSpGEMM on RMAT-16'
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 16 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 16 4 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 16 8 $OMP_NUM_THREADS 4096 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 16 16 $OMP_NUM_THREADS 16384 32

# run on r-mat scale 15
echo 'HeapSpGEMM on RMAT-15'
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 15 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 15 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 15 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 15 16 $OMP_NUM_THREADS 8192 32

# run on r-mat scale 14
echo 'HeapSpGEMM on RMAT-14'
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 14 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 14 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 14 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HeapSpGEMM_hw gen rmat 14 16 $OMP_NUM_THREADS 4096 32

#------
# Hash Algorithm
#------

# run on ER scale 20
echo 'HashSpGEMM on ER-20'
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 16 $OMP_NUM_THREADS 8192 16

# run on ER scale 19
echo 'HashSpGEMM on ER-19'
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 16 $OMP_NUM_THREADS 4096 16

# run on ER scale 18
echo 'HashSpGEMM on ER-18'
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 4 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 16 $OMP_NUM_THREADS 2048 32

# run on r-mat scale 16
echo 'HashSpGEMM on RMAT-16'
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 4 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 8 $OMP_NUM_THREADS 4096 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 16 $OMP_NUM_THREADS 16384 32

# run on r-mat scale 15
echo 'HashSpGEMM on RMAT-15'
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 16 $OMP_NUM_THREADS 8192 32

# run on r-mat scale 14
echo 'HashSpGEMM on RMAT-14'
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 16 $OMP_NUM_THREADS 4096 32

