# Masked Matrix-Matrix Multiplication (Masked SpGEMM)

The repository contains:
- Masked SpGEMM algorithm implementations: MSA, Hash, Heap, MCA
- Complemented mask variants: MSA, Hash, Heap
- Benchmarks: Triangle Counting, K-Truss, Betweenness Centrality

## Download repository

This project contains submodule, so use recursive option to download them all.

```shell
git clone --recursive https://github.com/PASSIONLab/MaskedSpGEMM.git
```
## Compile code

First compile dependencies:
```shell
cd MaskedSpGEMM/
make sprng rmat prmat
```

Then compile GraphBLAS:
```shell
cd GraphBLAS/
cd build
CC=gcc cmake ..
make -j
```

Compiling MaskedSpGEMM benchmarks:
```shell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j
```

## Running the benchmarks:
Algorithms and the number of iterations can be selected using environmental variables:
- MODE - set to benchmark to execute all algorithms
- WARMUP_ITERS - number of warmup iterations (not timed)
- INNER_ITERS - number of timed iterations

### Triangle counting
```shell
./build/tricnt-all-grb <path-to-mtx-file> <num-threads>
```
**Example**: Count the number of triangles for **belgium_osm.mtx** graph using **12 threads**
```shell
MODE=benchmark WARMUP_ITERS=3 INNER_ITERS=10 ./build/tricnt-all-grb belgium_osm.mtx 12
```

### K-Truss
```shell
./build/ktruss-all-grb <path-to-mtx-file> <k> <num-threads>
```
**Example**: Count the number of k-trusses (**k == 5**) for **belgium_osm.mtx** graph using **12 threads**
```shell
MODE=benchmark WARMUP_ITERS=3 INNER_ITERS=10 ./build/ktruss-all-grb belgium_osm.mtx 5 12
```

### Betweenness Centrality
```shell
./build/bc-all-grb <path-to-mtx-file> <batch-size> <num-threads>
```
**Example**: Calcuate Betweenness Centrality for a batch with **512 vertices** for **belgium_osm.mtx** graph using **12 threads**
```shell
MODE=benchmark WARMUP_ITERS=3 INNER_ITERS=10 ./build/bc-grb belgium_osm.mtx 5 12
```