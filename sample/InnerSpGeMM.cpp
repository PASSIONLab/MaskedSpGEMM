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
#include "../hash_mult_hw.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int
#define ITERS 10

int main(int argc, char* argv[])
{
    const bool sortOutput = false;
    vector<int> tnums;
    CSC<INDEXTYPE, VALUETYPE> A_csc, B_csc;


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
    SetInputMatricesAsCSC(A_csc, B_csc, argv); // Reading input mat and construct CSC
 
    CSR<INDEXTYPE, VALUETYPE> B_csr(B_csc); //converts, allocates and populates
    CSR<INDEXTYPE, VALUETYPE> A_csr(A_csc); //converts, allocates and populates
    
    A_csc.Sorted();
    A_csr.Sorted();
    B_csc.Sorted();
    B_csr.Sorted();

    A_csc.shuffleIds();
    B_csr.shuffleIds();

    /* Count total number of floating-point operations */
    long long int nfop = 1;//get_flop(A_csr, B_csc);
    // cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): " << nfop << endl << endl;

    double start, end, msec, ave_msec, mflops;

    /* Execute Hash-SpGEMM */
    cout << "Evaluation of InnerSpGEMM" << endl;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);

        CSR<INDEXTYPE,VALUETYPE> C_csr;
        CSR<INDEXTYPE,VALUETYPE> C_csr_tmp; //correctness check

        /*corrctness check */
        HashSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr_tmp, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        
        cout << "Hash SpGEMM wo Mask " << endl;
        for (int i = 0; i < 10; ++i)
        	cout << C_csr_tmp.values[i] << " ";
        cout << endl;
        C_csr_tmp.make_empty();
        
        /* First execution is excluded from evaluation */
        innerSpGEMM_nohash<false, sortOutput>(A_csr, B_csc, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        
        cout << "Dot SpGEMM with Mask" << endl;
        for (int i = 0; i < 10; ++i)
        	cout << C_csr.values[i] << " ";
        cout << endl;
        C_csr.make_empty();

        ave_msec = 0;
        for (int i = 0; i < ITERS; ++i) {
            start = omp_get_wtime();
            innerSpGEMM_nohash<false, sortOutput>(A_csr, B_csc, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                C_csr.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;

        // printf("DotSpGEMM returned with %d nonzeros. Compression ratio is %f\n", C_csr.nnz, (float)(nfop / 2) / (float)(C_csr.nnz));
        printf("DotSpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", tnum, ave_msec, mflops);

        C_csr.make_empty();
    }
    A_csr.make_empty();
    B_csr.make_empty();
    A_csc.make_empty();
    B_csc.make_empty();

    return 0;
}

