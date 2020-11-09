
# run on ER scale 20
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 20 16 $OMP_NUM_THREADS 8192 16

# run on ER scale 19
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 19 16 $OMP_NUM_THREADS 4096 16

# run on ER scale 18
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 2 $OMP_NUM_THREADS 256 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 4 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen er 18 16 $OMP_NUM_THREADS 2048 32

# run on r-mat scale 16
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 4 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 8 $OMP_NUM_THREADS 4096 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 16 16 $OMP_NUM_THREADS 16384 32

# run on r-mat scale 15
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 2 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 8 $OMP_NUM_THREADS 2048 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 15 16 $OMP_NUM_THREADS 8192 32

# run on r-mat scale 14
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 2 $OMP_NUM_THREADS 512 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 4 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 8 $OMP_NUM_THREADS 1024 32
numactl --membind=0  ./bin/HashSpGEMM_hw gen rmat 14 16 $OMP_NUM_THREADS 4096 32

