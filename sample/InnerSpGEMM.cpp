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
#include "../CSR.h"
#include "../CSC.h"
#include "../multiply.h"

#include "../inner_mult.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int
#define ITERS 1

int main(int argc, char* argv[])
{
    const bool sortOutput = true;
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
    B_csc.Sorted();
    M_csr.Sorted();

    /* Count total number of floating-point operations */
    long long int nfop = get_flop(A_csr, B_csr);
    cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * A): " << nfop << endl << endl;


    double start, end, msec, ave_msec, mflops;

    /* Execute Hash-SpGEMM */
    cout << "Evaluation of InnerSpGEMM" << endl;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);

        CSR<INDEXTYPE,VALUETYPE> C_csr;
        
        /* First execution is excluded from evaluation */
        innerSpGEMM_nohash<false, sortOutput>(M_csr, A_csr, B_csc, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        C_csr.make_empty();
   
        ave_msec = 0;
        for (int i = 0; i < ITERS; ++i) {
            start = omp_get_wtime();
            innerSpGEMM_nohash<false, sortOutput>(M_csr, A_csr, B_csc, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                C_csr.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;

        printf("DotSpGEMM returned with %d nonzeros. Compression ratio is %f\n", C_csr.nnz, (float)(nfop / 2) / (float)(C_csr.nnz));
        printf("DotSpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", tnum, ave_msec, mflops);

        C_csr.make_empty();
    }
    A_csr.make_empty();
    B_csr.make_empty();
    A_csc.make_empty();
    B_csc.make_empty();

    return 0;
}

