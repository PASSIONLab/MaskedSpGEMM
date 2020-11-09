#!/bin/bash

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NESTED=False
export OMP_MAX_ACTIVE_LEVELS=2


gcc -fopenmp -O3 -DSTREAM_ARRAY_SIZE=8000000 stream.c -o stream
numactl --membind=0 ./stream

