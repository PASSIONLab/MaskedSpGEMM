#!/usr/bin/env bash

for SCALE in 1 2 4 8
do
    exec ./bin/GenMatrices_hw gen rmat 12 $SCALE 48 256 128
done
