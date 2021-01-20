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
#include "../inner_mult.h"
#include "../mask_hash_mult.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int64_t
#define ITERS 4

int main(int argc, char* argv[])
{
    const bool sortOutput = false;

	if (argc < 4) {
        cout << "Normal usage: ./all_tc directory_name <gridx> <gridy>" << endl;
        cout << "All files should be under directory_name with names splitmatrix_gridx_gridy.mtx" << endl;
        return -1;
    }
	else {
        cout << "Running on " << argv[2] << " by " << argv[3] << "blocks" << endl << endl;
    }
    
    int gridx = atoi(argv[2]);
    int gridy = atoi(argv[3]);
    assert((gridx == gridy));
    
    string inputheader = string(argv[1])+"/splitmatrix_";
    vector< vector< CSC<INDEXTYPE,VALUETYPE> > > submatricesCSC(gridx);
    vector< vector< CSR<INDEXTYPE,VALUETYPE> > > submatricesCSR(gridx);
    
    /* Count total number of floating-point operations */
    vector< vector< vector<long long int> > >  flps(gridx);
    vector< vector< vector<long long int> > >  nnzs(gridx);
    vector< vector< vector<long long int> > >  masked_nnzs(gridx);
    
    vector< vector< vector<double> > >  timehash(gridx);
    vector< vector< vector<double> > >  timeinner(gridx);
    vector< vector< vector<double> > >  timemaskedspa(gridx);
    vector< vector< vector<double> > >  timemaskedhash(gridx);

    
    for (int i = 0; i< gridx; ++i)
    {
        flps[i].resize(gridy);
        nnzs[i].resize(gridy);
        masked_nnzs[i].resize(gridy);
        timehash[i].resize(gridy);
        timeinner[i].resize(gridy);
        timemaskedspa[i].resize(gridy);
        timemaskedhash[i].resize(gridy);

        for (int j = 0; j< gridy; ++j)
        {
            flps[i][j].resize(gridy);
            nnzs[i][j].resize(gridy);
            masked_nnzs[i][j].resize(gridy);
            timehash[i][j].resize(gridy);
            timeinner[i][j].resize(gridy);
            timemaskedspa[i][j].resize(gridy);
            timemaskedhash[i][j].resize(gridy);


            string inputfooter = to_string(i+1)+"_"+to_string(j+1)+".mtx";
            string inputname = inputheader + inputfooter;
            cout << "Reading file " << inputname << endl;
            CSC<INDEXTYPE,VALUETYPE> A_csc;
            ReadASCII(inputname, A_csc);
            submatricesCSC[i].push_back(A_csc);
            CSR<INDEXTYPE, VALUETYPE> A_csr(A_csc); //converts, allocates and populates
            submatricesCSR[i].push_back(A_csr);
            A_csc.Sorted();
            A_csr.Sorted();
        }
    }
    
    #pragma omp parallel for collapse(3)
    for (int i = 0; i< gridx; ++i)
    {
        for (int j = 0; j< gridy; ++j)
        {
            for (int k = 0; k< gridy; ++k)
            {
                flps[i][j][k] = get_flop(submatricesCSR[i][k], submatricesCSR[k][j]);
                #pragma omp critical
                {
                    printf("Thread %d out of %d: flps[%d][%d][%d]: %lld\n", omp_get_thread_num(), omp_get_num_threads(), i, j, k, flps[i][j][k]);
                }

                
                CSR<INDEXTYPE,VALUETYPE> C_csr;
                HashSpGEMM<false, sortOutput>(submatricesCSR[i][k], submatricesCSR[k][j], C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
                nnzs[i][j][k] = C_csr.nnz;
                
                
                C_csr.sortIds();
                CSR<INDEXTYPE,VALUETYPE> A_csr_sorted(submatricesCSR[i][j]);
                A_csr_sorted.sortIds();
                #pragma omp critical
                {
                    printf("Thread %d: HashSpGEMM executed and returned %lld nonzeros. Both matrices are sorted now\n", omp_get_thread_num(), C_csr.nnz);
                }

                CSR<INDEXTYPE,VALUETYPE> Tr_csr = Intersect(A_csr_sorted, C_csr, plus<VALUETYPE>());   // change plus to select2nd
                
                masked_nnzs[i][j][k] = Tr_csr.nnz;
                
                C_csr.make_empty();
                
                // A,B,C, Mask
                SPASpGEMM(submatricesCSR[i][k], submatricesCSR[k][j], C_csr, submatricesCSR[i][j], multiplies<VALUETYPE>(), plus<VALUETYPE>());
                #pragma omp critical
                {
                    printf("Thread %d: SPASpGEMM runs and returns %lld nonzeros\n",  omp_get_thread_num(), C_csr.nnz);
                }
                
                C_csr.make_empty();
                
                // Mask, A,B[csc],C
                innerSpGEMM_nohash<false, sortOutput>(submatricesCSR[i][j], submatricesCSR[i][k], submatricesCSC[k][j], C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());

                #pragma omp critical
                {
                    printf("Thread %d: innerSpGEMM_nohash runs and returns %lld nonzeros\n",  omp_get_thread_num(), C_csr.nnz);
                }
                C_csr.make_empty();
                
                // A,B,C, Mask
                mxm_hash_mask(submatricesCSR[i][k], submatricesCSR[k][j], C_csr, submatricesCSR[i][j], multiplies<VALUETYPE>(), plus<VALUETYPE>());
                #pragma omp critical
                {
                    printf("Thread %d: mxm_hash_mask runs and returns %lld nonzeros\n",  omp_get_thread_num(), C_csr.nnz);
                }
                
                timehash[i][j][k] = 0.0;
                timeinner[i][j][k] = 0.0;
                timemaskedspa[i][j][k] = 0.0;
                timemaskedhash[i][j][k] = 0.0;

                C_csr.make_empty();
            }
        }
    }
    
    long long int totalflops = 0;
    long long int totalnnzs = 0;
    long long int totalmaskednnz = 0;


    for (int i = 0; i< gridx; ++i)
    {
        for (int j = 0; j< gridy; ++j)
        {
            for (int k = 0; k< gridy; ++k)
            {
                totalflops += flps[i][j][k];
                totalnnzs += nnzs[i][j][k];
                totalmaskednnz += masked_nnzs[i][j][k];
            }
        }
    }

    cout << "Total number of floating-point operations in SpGEMM (A * A): " << totalflops << endl << endl;
    cout << "Total number of masked nnz outputs in SpGEMM (A * A): " << totalmaskednnz << endl << endl;

    
    
    double start, end, msec, mflops;

    /* Execute Hash-SpGEMM */
    cout << "Evaluation of HashSpGEMM (unsorted input/output)" << endl;
    start = omp_get_wtime();
    for (int i = 0; i < ITERS; ++i)
    {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i< gridx; ++i)
        {
            for (int j = 0; j< gridy; ++j)
            {
                for (int k = 0; k< gridy; ++k)
                {
                    CSR<INDEXTYPE,VALUETYPE> C_csr;
                    double localstart  = omp_get_wtime();
                    
                    HashSpGEMM<false, sortOutput>(submatricesCSR[i][k], submatricesCSR[k][j], C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
                    double localend = omp_get_wtime();

                    timehash[i][j][k] += (localend - localstart) * 1000;
                }
            }
        }
    }
    end = omp_get_wtime();
    msec = (end - start) * 1000;
    msec /= ITERS;

    mflops = (double)totalflops / msec / 1000;
    printf("HashSpGEMM returned with %d nonzeros. Compression ratio is %f\n", totalnnzs, (float)(totalflops / 2) / (float)(totalnnzs));
    printf("HashSpGEMM computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", msec, mflops);

    /* Execute Masked-SPA-SpGEMM */
    cout << "Evaluation of Masked SPA SpGEMM" << endl;
    start = omp_get_wtime();
    for (int i = 0; i < ITERS; ++i)
    {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i< gridx; ++i)
        {
            for (int j = 0; j< gridy; ++j)
            {
                for (int k = 0; k< gridy; ++k)
                {
                    CSR<INDEXTYPE,VALUETYPE> C_csr;
                    double localstart  = omp_get_wtime();
                       
                    SPASpGEMM(submatricesCSR[i][k], submatricesCSR[k][j], C_csr, submatricesCSR[i][j], multiplies<VALUETYPE>(), plus<VALUETYPE>());
                    double localend = omp_get_wtime();

                    timemaskedspa[i][j][k] += (localend - localstart) * 1000;
                }
            }
        }
    }
    end = omp_get_wtime();
    msec = (end - start) * 1000;
    msec /= ITERS;

    mflops = (double)totalflops / msec / 1000;
    printf("MaskedSPASpGEMM returned with %d nonzeros. Compression ratio is %f\n", totalnnzs, (float)(totalflops / 2) / (float)(totalmaskednnz));
    printf("MaskedSPASpGEMM computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", msec, mflops);
    
    
    /* Execute Inner Product SpGEMM */
    cout << "Evaluation of Inner Product SpGEMM" << endl;
    start = omp_get_wtime();
    for (int i = 0; i < ITERS; ++i)
    {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i< gridx; ++i)
        {
            for (int j = 0; j< gridy; ++j)
            {
                for (int k = 0; k< gridy; ++k)
                {
                    CSR<INDEXTYPE,VALUETYPE> C_csr;
                    double localstart  = omp_get_wtime();
                       
                    SPASpGEMM(submatricesCSR[i][k], submatricesCSR[k][j], C_csr, submatricesCSR[i][j], multiplies<VALUETYPE>(), plus<VALUETYPE>());
                    double localend = omp_get_wtime();

                    timemaskedspa[i][j][k] += (localend - localstart) * 1000;
                }
            }
        }
    }
    end = omp_get_wtime();
    msec = (end - start) * 1000;
    msec /= ITERS;

    mflops = (double)totalflops / msec / 1000;
    printf("MaskedSPASpGEMM returned with %d nonzeros. Compression ratio is %f\n", totalnnzs, (float)(totalflops / 2) / (float)(totalmaskednnz));
    printf("MaskedSPASpGEMM computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", msec, mflops);

    return 0;
}

