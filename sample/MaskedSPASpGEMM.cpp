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

#include "../utility.h"
#include "../CSR.h"
#include "../multiply.h"


#include "../hash_mult_hw.h"
#include "../spa_mult.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int
#define ITERS 10

int main(int argc, char* argv[])
{
    const bool sortOutput = false;
    vector<int> tnums;
	CSR<INDEXTYPE,VALUETYPE> A_csr, B_csr;

	if (argc < 4) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
        return -1;
    }
    else if (argc < 6) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;

#ifdef KNL_EXE
        cout << "Running on 68, 136, 204, 272 threads" << endl << endl;
        tnums = {68, 136, 204, 272};
        // tnums = {1, 2, 4, 8, 16, 32, 64, 68, 128, 136, 192, 204, 256, 272}; // for scalability test
#else
        cout << "Running on 32, 64 threads" << endl;
        tnums = {32, 64}; // for hashwell
#endif
    }
	else {
        cout << "Running on " << argv[5] << " processors" << endl << endl;
        tnums = {atoi(argv[5])};
    }

    /* Generating input matrices based on argument */
    SetInputMatricesAsCSR(A_csr, B_csr, argv);

  	A_csr.Sorted();
  	B_csr.Sorted();

    A_csr.shuffleIds();
    B_csr.shuffleIds();
    
    B_csr = A_csr;  // try the C = A^2 *. A triangle counting case

    /* Count total number of floating-point operations */
    long long int nfop = get_flop(A_csr, B_csr);
    cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * A): " << nfop << endl << endl;

    double start, end, msec, ave_msec, mflops;

    /* Execute Hash-SpGEMM */
    cout << "Evaluation of HashSpGEMM (unsorted input/output)" << endl;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);

        CSR<INDEXTYPE,VALUETYPE> C_csr;

        /* First execution is excluded from evaluation */
        HashSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        for (int i = 0; i < 10; ++i)
            cout << C_csr.values[i] << " ";
        cout << endl;
        C_csr.make_empty();

        ave_msec = 0;
        for (int i = 0; i < ITERS; ++i) {
            start = omp_get_wtime();
            HashSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                C_csr.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;

        printf("HashSpGEMM returned with %d nonzeros. Compression ratio is %f\n", C_csr.nnz, (float)(nfop / 2) / (float)(C_csr.nnz));
        printf("HashSpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", tnum, ave_msec, mflops);

        C_csr.sortIds();
        CSR<INDEXTYPE,VALUETYPE> A_csr_sorted(A_csr);
        A_csr.sortIds();
        printf("Both matrices are sorted now\n");

        
        CSR<INDEXTYPE,VALUETYPE> Tr_csr = Intersect(A_csr, C_csr, plus<VALUETYPE>());   // change plus to select2nd
        
        printf("A^2 *. A has %d nonzeros\n", Tr_csr.nnz);
        C_csr.make_empty();
    }

    /* Execute SPA-SpGEMM */
    cout << "Evaluation of MaskedSPASpGEMM (unsorted input/output)" << endl;
    ave_msec = 0;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);

        CSR<INDEXTYPE,VALUETYPE> C_csr;

        /* First execution is excluded from evaluation */
        /* Use A itself as the mask (4th parameter) */
        SPASpGEMM(A_csr, B_csr, C_csr, A_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        C_csr.make_empty();

        ave_msec = 0;
        for (int i = 0; i < ITERS; ++i) {
            start = omp_get_wtime();
            
            /* Use A itself as the mask (4th parameter) */
            SPASpGEMM(A_csr, B_csr, C_csr, A_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                C_csr.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;

        printf("MaskedSPASpGEMM returned with %d nonzeros. Compression ratio is %f\n", C_csr.nnz, (float)(nfop / 2) / (float)(C_csr.nnz));
        printf("MaskedSPASpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", tnum, ave_msec, mflops);

        C_csr.make_empty();
    }
    A_csr.make_empty();
    B_csr.make_empty();

    return 0;
}

