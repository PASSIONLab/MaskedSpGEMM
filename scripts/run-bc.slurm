#!/bin/bash -l

#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH --account m1982
#SBATCH -N 1
#SBATCH -J spgemm
#SBATCH --exclusive
#SBATCH --contiguous


while read l; do
  vs=($l);
  m=${vs[0]};
  export DISABLE_HEAP=1
  export OUTER_ITERS=1
  export WARMUP_ITERS=${vs[1]}
  export INNER_ITERS=${vs[2]}
  export MODE=benchmark

  if [ $NTHREADS -eq 32 ]; then
    export OMP_NUM_THREADS=32
    export GOMP_CPU_AFFINITY=0-63
  elif [ $NTHREADS -eq 68 ]; then
    export OMP_NUM_THREADS=68
    export GOMP_CPU_AFFINITY=0-271
  else
    exit 1
  fi;

  $APP_PATH $m ${vs[2]} $NTHREADS;
done < $INPUT_PATH