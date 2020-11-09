#!/bin/bash

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NESTED=False
export OMP_MAX_ACTIVE_LEVELS=2

#------
# OuterSpGEMM Algorithm
echo 'Running OuterSpGEMM experiments'
#------

numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/2cubes_sphere.mtx ./assets/2cubes_sphere.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/amazon0505.mtx ./assets/amazon0505.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/cage12.mtx ./assets/cage12.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/m133-b3.mtx ./assets/m133-b3.mtx ~/product.txt $OMP_NUM_THREADS 2048 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/majorbasis.mtx ./assets/majorbasis.mtx ~/product.txt $OMP_NUM_THREADS 2048 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/mc2depi.mtx ./assets/mc2depi.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/offshore.mtx ./assets/offshore.mtx ~/product.txt $OMP_NUM_THREADS 1024 64
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/patents_main.mtx ./assets/patents_main.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/scircuit.mtx ./assets/scircuit.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/OuterSpGEMM_hw text ./assets/web-Google.mtx ./assets/web-Google.mtx ~/product.txt $OMP_NUM_THREADS 1024 32

#------
# HashSpGEMM Algorithm
echo 'Running HashSpGEMM experiments'
#------

numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/2cubes_sphere.mtx ./assets/2cubes_sphere.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/amazon0505.mtx ./assets/amazon0505.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/cage12.mtx ./assets/cage12.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/m133-b3.mtx ./assets/m133-b3.mtx ~/product.txt $OMP_NUM_THREADS 2048 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/majorbasis.mtx ./assets/majorbasis.mtx ~/product.txt $OMP_NUM_THREADS 2048 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/mc2depi.mtx ./assets/mc2depi.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/offshore.mtx ./assets/offshore.mtx ~/product.txt $OMP_NUM_THREADS 1024 64
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/patents_main.mtx ./assets/patents_main.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/scircuit.mtx ./assets/scircuit.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HashSpGEMM_hw text ./assets/web-Google.mtx ./assets/web-Google.mtx ~/product.txt $OMP_NUM_THREADS 1024 32

#------
# HeapSpGEMM Algorithm
echo 'Running HeapSpGEMM experiments'
#------

numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/2cubes_sphere.mtx ./assets/2cubes_sphere.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/amazon0505.mtx ./assets/amazon0505.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/cage12.mtx ./assets/cage12.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/m133-b3.mtx ./assets/m133-b3.mtx ~/product.txt $OMP_NUM_THREADS 2048 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/majorbasis.mtx ./assets/majorbasis.mtx ~/product.txt $OMP_NUM_THREADS 2048 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/mc2depi.mtx ./assets/mc2depi.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/offshore.mtx ./assets/offshore.mtx ~/product.txt $OMP_NUM_THREADS 1024 64
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/patents_main.mtx ./assets/patents_main.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/scircuit.mtx ./assets/scircuit.mtx ~/product.txt $OMP_NUM_THREADS 1024 32
numactl --membind=0 ./bin/HeapSpGEMM_hw text ./assets/web-Google.mtx ./assets/web-Google.mtx ~/product.txt $OMP_NUM_THREADS 1024 32

