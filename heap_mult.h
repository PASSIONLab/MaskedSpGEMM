#include "CSC.h"
#include "utility.h"
#include <omp.h>
#include <algorithm>
#include <iostream>
using namespace std;

/**
 ** Count flop of SpGEMM between A and B in CSC format
 **/
template <typename IT, typename NT>
long long int get_flop(const CSC<IT,NT> & A, const CSC<IT,NT> & B, IT *maxnnzc)
{
    long long int flop = 0; // total flop (multiplication) needed to generate C
#pragma omp parallel
    {
        long long int tflop=0; //thread private flop
#pragma omp for
        for (IT i=0; i < B.cols; ++i) {       // for all columns of B
            long long int locmax = 0;
            for (IT j = B.colptr[i]; j < B.colptr[i+1]; ++j) {   // For all the nonzeros of the ith column
                IT inner = B.rowids[j];				// get the row id of B (or column id of A)
                IT npins = A.colptr[inner+1] - A.colptr[inner];	// get the number of nonzeros in A's corresponding column
                locmax += npins;
            }
            maxnnzc[i] = locmax;
            tflop += locmax;
        }
#pragma omp critical
        {
            flop += tflop;
        }
    }
    return flop * 2;
}

template <typename IT, typename NT>
long long int get_flop(const CSC<IT,NT> & A, const CSC<IT,NT> & B)
{
    IT *dummy = my_malloc<IT>(B.cols);
    long long int flop = get_flop(A, B, dummy);
    my_free<IT>(dummy);
    return flop;
}

template <typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HeapSpGEMM(const CSC<IT,NT> & A, const CSC<IT,NT> & B, CSC<IT,NT> & C, MultiplyOperation multop, AddOperation addop)
{
    int numThreads;
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }

    // *************** Load-balancing Thread Scheduling *********************
    IT *maxnnzc = my_malloc<IT>(B.cols);
    long long int flops = get_flop(A, B, maxnnzc) / 2;

    IT flopsPerThread = flops/numThreads; // amount of work that will be assigned to each thread

    IT *colPerThread = my_malloc<IT>(numThreads + 1); //thread i will process columns from colPerThread[i] to colPerThread[i+1]-1
    IT *colStart = my_malloc<IT>(B.cols); //start index in the global array for storing ith column of C
    IT *colEnd = my_malloc<IT>(B.cols); //end index in the global array for storing ith column of C

    colStart[0] = 0;
    colEnd[0] = 0;

    int curThread = 0;
    colPerThread[curThread++] = 0;
    IT nextflops = flopsPerThread;

    /* Parallelized version */
    scan(maxnnzc, colStart, B.cols);
#pragma omp parallel for
    for (int i = 1; i < B.cols; ++i) {
        colEnd[i] = colStart[i];
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long end_itr = (lower_bound(colStart, colStart + B.cols, flopsPerThread * (tid + 1))) - colStart;
        colPerThread[tid + 1] = end_itr;
    }
    colPerThread[numThreads] = B.cols;

    // *************** Creating global space to store result, used by all threads *********************
    IT size = colEnd[B.cols-1] + maxnnzc[B.cols-1];
    IT **LocalRowIdsofC = my_malloc<IT*>(numThreads);
    NT **LocalValuesofC = my_malloc<NT*>(numThreads);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        IT localsum = 0;
        for (IT i = colPerThread[tid]; i < colPerThread[tid + 1]; ++i) {
            localsum += maxnnzc[i];
        }
        LocalRowIdsofC[tid] = my_malloc<IT>(localsum);
        LocalValuesofC[tid] = my_malloc<NT>(localsum);
    }

    my_free<IT>(maxnnzc);

    // *************** Creating LOCAL heap space to be used by all threads *********************

    IT *threadHeapSize = my_malloc<IT>(numThreads);

#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        // IT localmax = -1; //incorrect
        IT localmax = 0;
        for (IT i = colPerThread[thisThread]; i < colPerThread[thisThread + 1]; ++i) {
            IT colnnz = B.colptr[i + 1] - B.colptr[i];
            if (colnnz > localmax)
                localmax = colnnz;
        }
        threadHeapSize[thisThread] = localmax;
    }

    // ************************ Numeric Phase *************************************
#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        HeapEntry<IT, NT> *mergeheap = my_malloc<HeapEntry<IT, NT>>(threadHeapSize[thisThread]);

        for (IT i = colPerThread[thisThread]; i < colPerThread[thisThread + 1]; ++i) {
            IT k = 0;   // Make initial heap
            for (IT j = B.colptr[i]; j < B.colptr[i + 1]; ++j) {   // For all the nonzeros of the ith column
                IT inner = B.rowids[j];				// get the row id of B (or column id of A)
                IT npins = A.colptr[inner + 1] - A.colptr[inner];	// get the number of nonzeros in A's corresponding column
                if (npins > 0) {
                    mergeheap[k].loc = 1;
                    mergeheap[k].runr = j;    			// the pointer to B.rowid's is the run-rank
                    mergeheap[k].value = multop(A.values[A.colptr[inner]], B.values[j]);
                    mergeheap[k++].key = A.rowids[A.colptr[inner]];	// A's first rowid is the first key
                }
            }
            IT hsize = k;      // if any of A's "significant" columns is empty, k will be less than hsize
            make_heap(mergeheap, mergeheap + hsize);

            while(hsize > 0) {
                pop_heap(mergeheap, mergeheap + hsize);         // result is stored in mergeheap[hsize-1]
                HeapEntry<IT,NT> hentry = mergeheap[hsize - 1];

                // Use short circuiting
                if ((colEnd[i] > colStart[i]) && LocalRowIdsofC[thisThread][colEnd[i] - colStart[colPerThread[thisThread]] - 1] == hentry.key) {
                    LocalValuesofC[thisThread][colEnd[i] - colStart[colPerThread[thisThread]] - 1] = addop(hentry.value, LocalValuesofC[thisThread][colEnd[i] - colStart[colPerThread[thisThread]] - 1]);
                }
                else {
                    LocalValuesofC[thisThread][colEnd[i] - colStart[colPerThread[thisThread]]]= hentry.value;
                    LocalRowIdsofC[thisThread][colEnd[i] - colStart[colPerThread[thisThread]]]= hentry.key;
                    colEnd[i] ++;
                }

                IT inner = B.rowids[hentry.runr];

                // If still unused nonzeros exists in A(:,colind), insert the next nonzero to the heap
                if ((A.colptr[inner + 1] - A.colptr[inner]) > hentry.loc) {
                    IT index = A.colptr[inner] + hentry.loc;
                    mergeheap[hsize-1].loc = hentry.loc + 1;
                    mergeheap[hsize-1].runr = hentry.runr;
                    mergeheap[hsize-1].value = multop(A.values[index], B.values[hentry.runr]);
                    mergeheap[hsize-1].key = A.rowids[index];
                    push_heap(mergeheap, mergeheap + hsize);
                }
                else {
                    --hsize;
                }
            }
        }
        my_free<HeapEntry<IT, NT>>(mergeheap);
    }

    my_free<IT>(threadHeapSize);

    if (C.isEmpty()) {
        C.make_empty();
    }

    // ************************ Copy output to C *************************************
    C.rows = A.rows;
    C.cols = B.cols;

    C.colptr = my_malloc<IT>(C.cols + 1);
    C.colptr[0] = 0;

    IT *col_nz = my_malloc<IT>(C.cols);
#pragma omp parallel for
    for (int i = 0; i < C.cols; ++i) {
        col_nz[i] = colEnd[i] - colStart[i];
    }
    scan(col_nz, C.colptr, C.cols + 1);
    my_free<IT>(col_nz);

    C.nnz = C.colptr[C.cols];
    C.rowids = my_malloc<IT>(C.nnz);
    C.values = my_malloc<NT>(C.nnz);

#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        for(int i = colPerThread[thisThread]; i< colPerThread[thisThread + 1]; ++i) {        // combine step
            copy(&LocalRowIdsofC[thisThread][colStart[i] - colStart[colPerThread[thisThread]]], &LocalRowIdsofC[thisThread][colEnd[i] - colStart[colPerThread[thisThread]]], C.rowids + C.colptr[i]);
            copy(&LocalValuesofC[thisThread][colStart[i] - colStart[colPerThread[thisThread]]], &LocalValuesofC[thisThread][colEnd[i] - colStart[colPerThread[thisThread]]], C.values + C.colptr[i]);
        }
    }

    // ************************ Memory deallocation *************************************
#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        my_free<IT>(LocalRowIdsofC[thisThread]);
        my_free<NT>(LocalValuesofC[thisThread]);
    }
    my_free<IT*>(LocalRowIdsofC);
    my_free<NT*>(LocalValuesofC);
    my_free<IT>(colPerThread);
    my_free<IT>(colEnd);
    my_free<IT>(colStart);

}

