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
#define ITERS 1

int main(int argc, char* argv[])
{
    const bool sortOutput = false;
    vector<int> tnums;

	if (argc < 4) {
        cout << "Normal usage: ./spgemm matrix1.mtx matrix2.mtx mask.mtx <numthreads>" << endl;
        return -1;
    }
    else if (argc < 5) {
        cout << "Normal usage: ./spgemm matrix1.mtx matrix2.mtx mask.mtx <numthreads>" << endl;

#ifdef KNL_EXE
        cout << "Running on 68, 136, 204, 272 threads" << endl << endl;
        tnums = {68, 136, 204, 272};
        // tnums = {1, 2, 4, 8, 16, 32, 64, 68, 128, 136, 192, 204, 256, 272}; // for scalability test
#else
        cout << "Running on 4, 8, 16, 32, 64 threads" << endl;
        tnums = {4, 8, 16, 32, 64}; // for hashwell
#endif
    }
	else {
        cout << "Running on " << argv[4] << " processors" << endl << endl;
        tnums = {atoi(argv[4])};
    }

    string inputname1 = argv[1];
    CSC<INDEXTYPE,VALUETYPE> A_csc;
    ReadASCII(inputname1, A_csc);
    CSR<INDEXTYPE, VALUETYPE> A_csr(A_csc); //converts, allocates and populates

    string inputname2 = argv[2];
    CSC<INDEXTYPE,VALUETYPE> B_csc;
    ReadASCII(inputname2, B_csc);
    CSR<INDEXTYPE, VALUETYPE> B_csr(B_csc); //converts, allocates and populates
    
    string inputname3 = argv[3];
    CSC<INDEXTYPE,VALUETYPE> M_csc;
    ReadASCII(inputname3, M_csc);
    CSR<INDEXTYPE, VALUETYPE> M_csr(M_csc); //converts, allocates and populates

  	A_csr.Sorted();
  	B_csr.Sorted();
    M_csr.Sorted();

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
        
        CSR<INDEXTYPE,VALUETYPE> Tr_csr = Intersect(M_csr, C_csr, plus<VALUETYPE>());   // change plus to select2nd
        
        printf("AB *. M has %d nonzeros\n", Tr_csr.nnz);
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
        SPASpGEMM(A_csr, B_csr, C_csr, M_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        C_csr.make_empty();

        ave_msec = 0;
        for (int i = 0; i < ITERS; ++i) {
            start = omp_get_wtime();
            
            /* Use A itself as the mask (4th parameter) */
            SPASpGEMM(A_csr, B_csr, C_csr, M_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
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
    M_csr.make_empty();


    return 0;
}

