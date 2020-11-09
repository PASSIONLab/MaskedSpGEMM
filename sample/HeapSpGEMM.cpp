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
#include "../heap_mult.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int
#define ITERS 10

int main(int argc, char* argv[])
{
    vector<int> tnums;
	CSC<INDEXTYPE,VALUETYPE> A_csc, B_csc;

	if (argc < 4) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
        return -1;
    }
    else if (argc < 6) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;

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
    SetInputMatricesAsCSC(A_csc, B_csc, argv);

  	A_csc.Sorted();
  	B_csc.Sorted();

    /* Count total number of floating-point operations */
    long long int nfop = get_flop(A_csc, B_csc);
    cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): " << nfop << endl << endl;

    /* Execute SpGEMM C = A * B */
    double start, end, msec, ave_msec, mflops;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);

        CSC<INDEXTYPE,VALUETYPE> C_csc;

        /* First execution is excluded from evaluation */
        HeapSpGEMM(A_csc, B_csc, C_csc, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        C_csc.make_empty();

        // A_csc.shuffleIds();
        // B_csc.shuffleIds();
        ave_msec = 0;
        for (int i = 0; i < ITERS; ++i) {
            start = omp_get_wtime();
            HeapSpGEMM(A_csc, B_csc, C_csc, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                C_csc.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;

        printf("HeapSpGEMM returned with %d nonzeros. Compression ratio is %f\n", C_csc.nnz, (float)(nfop / 2) / (float)(C_csc.nnz));
        printf("HeapSpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", tnum, ave_msec, mflops);

        C_csc.make_empty();
    }

    A_csc.make_empty();
    B_csc.make_empty();

    return 0;
}

