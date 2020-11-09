#!/bin/bash

make clean && make spgemm
rm -rf result*.txt assets

for ITER in 1 2 3
do
    bash ./scripts/run_synthetic.sh > result_synthetic_$ITER.txt
done

bash ./scripts/download.sh

for ITER in 1 2 3
do
    bash ./scripts/run_real.sh > result_real_$ITER.txt
done

bash ./scripts/run_STREAM.sh > result_STREAM.txt

tar cvzf my_result.tar.gz result*.txt
rm -rf results*

