#include "CSR.h"
#include <omp.h>
#include <algorithm>
// #include <mkl_spblas.h>


template <bool sortOutput, typename IT>
int MKLSpGEMM_symbolic(const CSR<IT,float> &A, const CSR<IT,float> &B, CSR<IT,float> &C)
{
    int request = 1;
    int sort = 7; // don't sort anything
    int info = 0; // output info flag

    mkl_scsrmultcsr((char*)"N", &request, &sort,
                    &(A.rows), &(A.cols), &(B.cols),
                    A.values, A.colids, A.rowptr,
                    B.values, B.colids, B.rowptr,
                    NULL, NULL, C.rowptr,
                    NULL, &info);

    return info;
}

template <bool sortOutput, typename IT>
void MKLSpGEMM_symbolic(const CSR<IT,double> &A, const CSR<IT,double> &B, CSR<IT,double> &C)
{
    int request = 1;
    int sort = 7; // don't sort anything
    int info = 0; // output info flag

    mkl_dcsrmultcsr((char*)"N", &request, &sort,
                    &(A.rows), &(A.cols), &(B.cols),
                    A.values, A.colids, A.rowptr,
                    B.values, B.colids, B.rowptr,
                    NULL, NULL, C.rowptr,
                    NULL, &info);
}

template <bool sortOutput, typename IT>
int MKLSpGEMM_numeric(const CSR<IT,float> &A, const CSR<IT,float> &B, CSR<IT,float> &C)
{
    int request = 2;
    int sort = 7;
    int info = 0; // output info flag
    if (sortOutput) {
        sort = 8; // sort nonzeroes in rows of C, leave A and B alone (they are already sorted)
    }
    mkl_scsrmultcsr((char*)"N", &request, &sort,
                    &(A.rows), &(A.cols), &(B.cols),
                    A.values, A.colids, A.rowptr,
                    B.values, B.colids, B.rowptr,
                    C.values, C.colids, C.rowptr,
                    NULL, &info);
    return info;
}

template <bool sortOutput, typename IT>
int MKLSpGEMM_numeric(const CSR<IT,double> &A, const CSR<IT,double> &B, CSR<IT,double> &C)
{
    int request = 2;
    int sort = 7;
    int info = 0; // output info flag
    if (sortOutput) {
        sort = 8; // sort nonzeroes in rows of C, leave A and B alone (they are already sorted)
    }
    mkl_dcsrmultcsr((char*)"N", &request, &sort,
                    &(A.rows), &(A.cols), &(B.cols),
                    A.values, A.colids, A.rowptr,
                    B.values, B.colids, B.rowptr,
                    C.values, C.colids, C.rowptr,
                    NULL, &info);
    return info;
}

template <bool sortOutput, typename IT, typename NT>
void MKLSpGEMM(const CSR<IT,NT> &A, const CSR<IT,NT> &B, CSR<IT,NT> &C)
{
    // for request=1, mkl_dcsrmultcsr() computes only values of the array ic of length m + 1,
    // the memory for this array must be allocated beforehand. On exit the value
    // ic(m+1) - 1 is the actual number of the elements in the arrays c and jc

    int info;
    if (typeid(IT) != typeid(int)) {
        cout << "MKL does not support non-int type indices." << endl;
        return;
    }

    C.rows = A.rows;
    C.cols = B.cols;
    C.rowptr = my_malloc<IT>(C.rows + 1);
    C.zerobased = false;

    info = MKLSpGEMM_symbolic<sortOutput, IT>(A, B, C);

    if (info != 0) {
        cout << "MKL-Count Error: info returned " << info << endl;
        assert(info == 0);
    }

    C.nnz = C.rowptr[A.rows] - 1;

    C.colids = my_malloc<IT>(C.nnz);
    C.values = my_malloc<NT>(C.nnz);

    // for request=2, mkl_dcsrmultcsr() has been called previously with the parameter request=1,
    // the output arrays jc and c are allocated in the calling program and they are of the length ic(m+1) - 1 at least.

    info = MKLSpGEMM_numeric<sortOutput, IT>(A, B, C);

    if (info != 0) {
        printf("MKL-Calculation Error: info returned %d\n", info);
        assert(info == 0);
    }
}

template <typename IT, typename NT>
long long int get_flop(const CSR<IT,NT> & A, const CSR<IT,NT> & B)
{
    long long int flops = 0; // total flops (multiplication) needed to generate C
    long long int tflops=0; //thread private flops

    for (IT i=0; i < A.rows; ++i) {       // for all rows of A
        long long int locmax = 0;
        for (IT j=A.rowptr[i]; j < A.rowptr[i + 1]; ++j) { // For all the nonzeros of the ith column
            long long int inner = A.colids[j]; // get the row id of B (or column id of A)
            long long int npins = B.rowptr[inner + 1] - B.rowptr[inner]; // get the number of nonzeros in A's corresponding column
            locmax += npins;
        }
        tflops += locmax;
    }
    flops += tflops;
    return (flops * 2);
}


template <typename IT, typename NT>
long long int get_flop(const CSC<IT, NT> &A, const CSR<IT, NT> &B)
{
    long long int flops = 0;

#pragma omp parallel for reduction(+ \
                                   : flops)
    for (IT i = 0; i < A.cols; ++i)
    {
        IT colnnz = A.colptr[i + 1] - A.colptr[i];
        IT rownnz = B.rowptr[i + 1] - B.rowptr[i];
        flops += (colnnz * rownnz);
    }
    return (flops * 2);
}
