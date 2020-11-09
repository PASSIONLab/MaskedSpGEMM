#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>


#include "../utility.h"
#include "../CSC.h"
#include "sample_common.hpp"

using namespace std;


#define VALUETYPE double
#define INDEXTYPE int32_t


int main(int argc, char* argv[])
{

	CSC<INDEXTYPE, VALUETYPE> A_csc, B_csc;

	if (argc < 4) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
        return -1;
    }
    else if (argc < 6) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
    }
    else {
        cout << "Running on " << argv[5] << " processors" << endl << endl;
    }

    /* Generating input matrices based on argument */
    SetInputMatricesAsCSC(A_csc, B_csc, argv);
    CSR<INDEXTYPE, VALUETYPE> B_csr (B_csc);

    std::ofstream fd_left;
    fd_left.open(std::string("assets/left_") + std::string(argv[2]) + std::string(argv[3]) + std::string("_") + std::string(argv[4]) + std::string(".mtx"));
    fd_left << "%%MatrixMarket matrix coordinate pattern general" << endl;
    fd_left << A_csc.rows << " " << A_csc.cols << " " << A_csc.nnz << endl;
    for (int32_t i = 0; i < A_csc.rows; ++i)
        for (int32_t j = A_csc.colptr[i]; j < A_csc.colptr[i + 1]; ++j)
            fd_left << A_csc.rowids[j] + 1 << " " << i + 1 << std::endl;
    fd_left.close();

    std::ofstream fd_right;
    fd_right.open(std::string("assets/right_") + std::string(argv[2]) + std::string(argv[3]) + std::string("_") + std::string(argv[4]) + std::string(".mtx"));
    fd_right << "%%MatrixMarket matrix coordinate pattern general" << endl;
    fd_right << B_csr.rows << " " << B_csr.cols << " " << B_csr.nnz << endl;
    for (int32_t i = 0; i < B_csr.rows; ++i)
        for (int32_t j = B_csr.rowptr[i]; j < B_csr.rowptr[i + 1]; ++j)
            fd_right << i + 1 << " " << B_csr.colids[j] + 1 << std::endl;
    fd_right.close();

    printf("Left matrix has %d nonzeros, right matrix has %d nonzeros, nrows %d\n", A_csc.nnz, B_csr.nnz, A_csc.rows);

    A_csc.make_empty();
    B_csc.make_empty();
    B_csr.make_empty();
    return 0;
}
