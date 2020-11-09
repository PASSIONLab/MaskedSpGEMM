#!/usr/bin/env bash

for SCALE in 1 2 4 8
do
    ./bin/GenMatrices_hw gen er 23 $SCALE 48 256 128
done
