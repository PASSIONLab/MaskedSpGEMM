## Masked and Blocked SpGEMM (Sparse Matrix Matrix Multiplication)

- Implements Masked SpGEMM algorithms
- Implements a blocked SpGEMM as a gateway for dynamic algorithm selection and execution
- Also includes heap-, hash-, outer-product-, and inner-product based non-masked SpGEMM codes for comparison. 


## Download repository

This project contains submodule, so use recursive option to download them all.

```bash
git clone --recursive https://github.com/PASSIONLab/MaskedSpGEMM.git
```

## Compile code

First, compile GraphBLAS:
Make sure you use the right Intel compiler on cori (need v19.1):
```bash
module swap intel/19.0.3.199 intel/19.1.2.254
```
Then, to compile it:
```bash
cd GraphBLAS && mkdir build && cd build
CC=icc cmake ..
make --jobs=16
```
Then, compile the masked spgemm codes:
```bash
make clean && make spgemm
```

## Generate test matrices

Please make sure you have compiled the code before generating matrices.

For ER matrices, type the next command and we will generate four ER matrices for test, with fix scale(23) and edge factor {1, 2, 4, 8}, modify`scripts/gen-er.sh` if you want other parameters.

```bash
make gen-er
```

For R-MAT matrices, type the next command and we will generate four R-MAT matrices for test, with fix scale(10) and edge factor {1, 2, 4, 8}, modify`scripts/gen-rmat.sh` if you want other parameters.

```bash
make gen-rmat
```

All the test cases we will have a left matrix and a right matrix in the folder `assets/`, follow the naming pattern `left_{matrix_type}_{scale}_{edge}.mtx`.

## Download real world matrices

Anything with .mtx extension (matrix market format) should work, so should a simple triples representation


## Run sample programs

Here is the general instruction to run the top 2 performing masked SpGEMM codes.

```bash
./bin/MaskedSPASpGEMM_hw inputMatrix1.mtx inputMatrix2.mtx maskMatrix.mtx <num_threads>
./bin/InnerSpGEMM_hw inputMatrix1.mtx inputMatrix2.mtx maskMatrix.mtx <num_threads>
```

You can also run triangle counting with the blocked code as follows, which will compute L.* (L*L) for a given lower triangular part of the adjacency matrix

```bash
 ./bin/blocked_parallel_tc_hw <directory_path> <gridx> <gridy>

# run blocked_parallel_tc_hw webgoogle lower triangular matrix that was previously sorted by degree and split into 8 pieces
# this will run all algorithms on the data
 ./bin/blocked_parallel_tc_hw webgoogle_tril_degsorted-proc8-by-8 8 8
```

The codes for degree sorting to lower triangular matrices as well as splitting an input into p_x by p_y grid is available under the Matlab directory



## Clean up

Delete them all.

```bash
make clean
```
