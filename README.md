# OuterSpGEMM
Outer product Sparse Matrix Matrix Multiplication

![outer_illustration](images/outer_illustration.png)

## Download repository

This project contains submodule, so use recursive option to download them all.

```bash
git clone --recursive git@bitbucket.org:azadcse/outerspgemm.git
```

## Compile code

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


## Run sample programs

Here is the general Instruction.

```bash
./bin/OuterSpGEMM text {left_matrix} {right_matrix} {output_file} {nthreads} {nblockers} {block_width}
```

Some example that might help.

```bash

# run OuterGpGEMM on pre-generated ER(23, 4) matrix, 48 threads, 256 blockers and width of each blocker is 128
./bin/OuterSpGEMM_hw text assets/left_er23_4.mtx assets/right_er23_4.mtx ~/product.txt 48 256 128

# run OuterGpGEMM on run time ER(18, 18) matrix, 48 threads, 256 blockers and width of each blocker is 128
./bin/OuterSpGEMM_hw gen er 18 18 48 256 128

# profile data of OuterGpGEMM on run downloaded web-Google matrix, 48 threads, 256 blockers and width of each blocker is 128
./bin/ProfileOuterSpGEMM_hw text assets/web-Google.mtx assets/web-Google.mtx ~/product.txt 48 128 128

```

Using only one socket
```bash

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NESTED=False

// run the program
numactl --membind=0 ./bin/OuterSpGEMM_hw gen er 20 4 24 1024 32
```

Using two sockets(Performance is not good yet)
```bash
export OMP_PLACES=sockets
export OMP_PROC_BIND=spread,close
export OMP_NESTED=True
export OMP_MAX_ACTIVE_LEVELS=2

// run the program
numactl --localalloc ././bin/NUMA-aware_hw gen er 20 4 24 1024 32

```
## Clean up

Delete them all.

```bash
make clean
```
