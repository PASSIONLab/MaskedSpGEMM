#include <omp.h>
#include <stdio.h>
#include <iostream>
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


//#include "overridenew.h"
#include "../utility.h"
#include "../CSC.h"
#include "../multiply.h"
#include "../Triple.h"
#include "../heap_mult.h"
#include "sample_common.hpp"

using namespace std;

#define ITER 1
#define VALUETYPE double
#define INDEXTYPE int32_t

class M {
    public:
        int *a;
        int *b;
    M(int size) {
        a = my_malloc<int>(size);
        b = my_malloc<int>(size);
    };
};

template <typename TYPE>
void ReduceDynamicArray(TYPE* a , int size)
{
    TYPE sum = 0;
#pragma omp parallel for
    for (int j=0; j<size; j++)
    {
        a[j] = 1;
    }
    double start = omp_get_wtime();
    for(int i=0; i<ITER; i++)
    {
#pragma omp parallel for reduction(+ : sum)
        for (int j=0; j<size; j++)
            sum += a[j];
    }

    double end  = omp_get_wtime();
    double msec = ((end - start) * 1000)/ITER ;
    double bandwidth =  (double) size * sizeof(TYPE) / 1024 / 1024 / msec;
    cout << "StreamTest [ReduceDynamicArray] : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << " "<< sum << endl;
}

template <typename IT>
void ReadBW_inside(size_t size)
{
    IT dummySum = 0;
    // IT* inside = my_malloc<IT>(size);
    IT* inside = new IT[size];
#pragma omp parallel for
    for (auto i = 0; i < size; ++i) {
        inside[i] = i % 65535;
    }
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;

    t1 = omp_get_wtime();
    for (int j = 0; j < ITER; ++ j)
    {
#pragma omp parallel for reduction (+ : dummySum)
    for (auto i = 0; i < size; ++i) {
        dummySum += inside[i];
    }
    cout << "dummySum=" << dummySum << endl;
    }
    t2 = omp_get_wtime();
    t3 += (t2 - t1);

    double msec = (t3 * 1000)/ITER;
    double bandwidth =  (double) size * sizeof(IT) / 1024 / 1024 / 1024 / (msec / 1000);
    cout << "StreamTest [ReadBW_inside] : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << " "<< dummySum << endl;
}

template <typename IT>
void ReadBW_outside(IT* outside, size_t size)
{
    IT dummySum = 0;

#pragma omp parallel for
    for (auto i = 0; i < size; ++i) {
        outside[i] = i % 65535;
    }
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;

    t1 = omp_get_wtime();
    for (int j = 0; j < ITER; ++ j)
#pragma omp parallel for reduction(+ : dummySum)
    for (auto i = 0; i < size; ++i) {
        dummySum += outside[i];
    }
    t2 = omp_get_wtime();
    t3 += (t2 - t1);

    double msec = (t3 * 1000)/ITER;
    double bandwidth =  (double) size * sizeof(IT) / 1024 / 1024 / 1024 / (msec / 1000);
    cout << "StreamTest [ReadBW_outside]: " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << " "<< dummySum << endl;

}

template <typename IT, typename NT>
void ReadBW_OuterSpGEMM(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    typedef tuple<IT, NT> TripleNode;
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;
    double sum = 0.0;
        sum = 0;
    std::ofstream fd;
    fd.open("nonzeros_by_rows.csv");
// #pragma omp parallel for reduction(+:sum)
    for (int idx=0; idx<A.cols; idx++) {
        fd << idx << " " << A.colptr[idx+1] - A.colptr[idx] << " " << B.rowptr[idx+1] - B.rowptr[idx] << endl;
        for (int i=A.colptr[idx]; i<A.colptr[idx+1]; ++i) {
            for (int j=B.rowptr[idx]; j<B.rowptr[idx+1]; ++j) {
                // avoid false sharing
                sum += (A.rowids[i] + B.colids[j] + A.values[i] + B.values[j]);
                // local_sum += (A.rowids[i] + B.colids[j] + A.values[i] + B.values[j]);
            }
        }
    }
    fd.close();
    cout << "Total = " << sum << endl;

    t1 = omp_get_wtime();
    for (int g=0; g<ITER; ++g)
    {
        sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int idx=0; idx<A.cols; idx++) {
        for (int i=A.colptr[idx]; i<A.colptr[idx+1]; ++i) {
            for (int j=B.rowptr[idx]; j<B.rowptr[idx+1]; ++j) {
                // avoid false sharing
                sum += (A.rowids[i] + B.colids[j] + A.values[i] + B.values[j]);
                // local_sum += (A.rowids[i] + B.colids[j] + A.values[i] + B.values[j]);
            }
        }
    }
        cout << "Total = " << sum << endl;
    }
    t2 = omp_get_wtime();
    t3 += (t2 - t1);
    double flops = get_flop(A, B);

    // double bytes = (A.nnz + B.nnz) * (sizeof(IT) + sizeof(NT)) + (A.cols + B.rows) * sizeof(IT) + sizeof(IT) * get_flop(A, B);
    double gbytes = (double)((A.nnz + B.nnz) * (sizeof(IT) + sizeof(NT)) + (A.cols + B.rows) * sizeof(IT)) / (1024 * 1024 * 1024);
    cout << "outer gbytes = " << gbytes << endl;
    // double msec = (t3 / ITER  - 8 * flops / (1024 * 1024 * 1024) / 1953.58 ) * 1024;
    double msec = (t3 / ITER ) * 1000;
    double bandwidth =  gbytes / (msec / 1000);
    // cout << "Time removed = " << 2 * 64.0 * flops / 1000000000 / 1953.58 <<endl;
    cout << "StreamTest [ReadBW_OuterSpGEMM]: " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}

template <typename IT, typename NT>
void ReadBW_ColByColSpGEMM(const CSC<IT, NT>& A, const CSC<IT, NT>& B, int d)
{
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;
    double total = 0;
    double sum;

    sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i=0; i<B.cols; ++i) {
        for (int j=B.colptr[i]; j<B.colptr[i+1]; ++j) {
            int rowid = B.rowids[j];
            for (int k=A.colptr[rowid]; k<A.colptr[rowid+1]; ++k) {
                sum = (A.rowids[k] + B.rowids[j] + A.values[k] + B.values[j]);
            }
        }
    }
    cout << "Total = " << sum << std::endl;
    t1 = omp_get_wtime();

for (int iter=0; iter<ITER; ++iter)
{
    sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int i=0; i<B.cols; ++i) {
        for (int j=B.colptr[i]; j<B.colptr[i+1]; ++j) {
            int rowid = B.rowids[j];
            for (int k=A.colptr[rowid]; k<A.colptr[rowid+1]; ++k) {
                sum += (A.rowids[k] + B.rowids[j] + A.values[k] + B.values[j]);
            }
        }
    }
    cout << "Total = " << sum << std::endl;
}
    t2 = omp_get_wtime();
    t3 += (t2 - t1);
    double flops = get_flop(A, B);
    // double bytes = (A.nnz + B.nnz) * (sizeof(IT) + sizeof(NT)) + (A.cols + B.rows) * sizeof(IT) + sizeof(IT) * get_flop(A, B);
    double gbytes = (double)((d * A.nnz + B.nnz) * (sizeof(IT) + sizeof(NT)) + (d * A.cols + B.rows) * sizeof(IT)) / (1024 * 1024 * 1024);
    cout << "column gbytes = " << gbytes << " d = " << d << endl;
    double msec = (t3 * 1000)/ ITER;
    // double msec = (t3 / ITER  - 8 * flops / (1024 * 1024 * 1024) / 1953.58 ) * 1024;
    double bandwidth =  gbytes / (msec / 1000);
    cout << "StreamTest [ReadBW_ColByColSpGEMM]: " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}

template <typename IT, typename NT>
void mtxstream(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    IT* a             = A.rowids;
    IT* b             = B.colids;
    double start_time = omp_get_wtime();
    int niter         = 10;
    for (int iter = 0; iter < niter; ++iter)
    {
        for (IT i = 0; i < A.cols; ++i)  // outer product of ith row of A and ith column of B
        {
            // IT rownnz = B.rowptr[i+1] - B.rowptr[i];
            // IT colnnz = A.colptr[i+1] - A.colptr[i];
            // total_flop += (colnnz * rownnz);
            // total_flop += (B.rowptr[i] - A.colptr[i]);
            IT start = A.colptr[i];
            IT end   = A.colptr[i + 1];
          for (IT j = start; j < end; ++j)  // For all the nonzeros of the ith column
          {
              a[j] = b[j];
          }
      }
  }

    double end_time = omp_get_wtime();
    double msec     = ((end_time - start_time) * 1000) / niter;
    double N        = A.nnz + A.rows;
    int itemsize    = sizeof(IT);

    double bandwidth = 2 * (double) N * itemsize / 1000000 / msec;
    cout << "bandwidth : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}

template <typename NT>
void stream(vector<NT> a, vector<NT> b, int itemsize)
{
    int64_t N    = a.size();
    double start = omp_get_wtime();
    int niter    = 1000;


    for (int iter = 0; iter < niter; ++iter)
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            a[i] = b[i];
        }
    }
    double end  = omp_get_wtime();
    double msec = ((end - start) * 1000) / niter;

    double bandwidth = 2 * (double) N * itemsize / 1024 / 1024 / msec;
    cout << "StreamTest : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}

template <typename IT, typename NT>
void StreamTest(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    IT* a32 = new IT[A.nnz]();
    NT* a64 = new NT[A.nnz]();
    ReduceDynamicArray(a32, A.nnz);
    ReduceDynamicArray(a64, A.nnz);
    ReduceDynamicArray(A.values, A.nnz);
    ReduceDynamicArray(A.rowids, A.nnz);
    delete[] a32;
    delete[] a64;

    //vector<IT> stream1(A.nnz, 0);
    //std::iota(stream1.begin(), stream1.end(), 0);
    //vector<IT> stream2(A.nnz, 0);
    //std::iota(stream2.begin(), stream2.end(), 0);
    //stream(stream1, stream2, sizeof(IT));
}


int main(int argc, char* argv[])
{
    vector<int> tnums = {1};

	CSC<INDEXTYPE, VALUETYPE> A_csc, B_csc, C_csc_corret;


	if (argc < 4) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
        return -1;
    }
    else if (argc < 6) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
    }
    else {
        cout << "Running on " << argv[5] << " processors" << endl << endl;
        tnums = {atoi(argv[5])};
    }
    /* Generating input matrices based on argument */
    SetInputMatricesAsCSC(A_csc, B_csc, argv);

    CSR<INDEXTYPE, VALUETYPE> B_csr (B_csc);
    // CSR<INDEXTYPE, VALUETYPE> A_csr (A_csc);
    //   A_csc.Sorted();
    //   B_csc.Sorted();
    // A_csc.shuffleIds();
    // B_csc.shuffleIds();
    // B_csr.shuffleIds();
//    B_csr.Sorted();
    // M m = M(A_csc.nnz);
    INDEXTYPE* myOutside = my_malloc<INDEXTYPE>(A_csc.nnz);
    //
    cout << "start evaluation" << endl;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);
        // StreamTest(A_csc, B_csr);
        ReadBW_inside<INDEXTYPE>(A_csc.nnz);
        // ReadBW_outside<INDEXTYPE>(myOutside, A_csc.nnz);
        // ReadBW_outside<INDEXTYPE>(B_csr.colids, B_csr.nnz);
        // ReadBW_outside<INDEXTYPE>(B_csr.colids, B_csr.nnz);
        // ReadBW_outside<INDEXTYPE>(B_csc.rowids, B_csc.nnz);
        // ReadBW_outside<INDEXTYPE>(m.a, A_csc.nnz);
        // ReadBW_outside<INDEXTYPE>(m.b, A_csc.nnz);
        ReadBW_OuterSpGEMM<INDEXTYPE, VALUETYPE>(A_csc, B_csr);
        ReadBW_ColByColSpGEMM<INDEXTYPE, VALUETYPE>(A_csc, B_csc, (int)atoi(argv[4]));
    }

    A_csc.make_empty();
    B_csc.make_empty();
 //   B_csr.make_empty();

    return 0;
}
