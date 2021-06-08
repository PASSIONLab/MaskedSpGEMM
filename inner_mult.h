#include "CSC.h"
#include "utility.h"
#include "hash_mult_hw.h"
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
                IT inner = B.rowids[j];             // get the row id of B (or column id of A)
                IT npins = A.colptr[inner+1] - A.colptr[inner]; // get the number of nonzeros in A's corresponding column
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


//*TODO:: Dealing with 5 mats. Mask, A, B, C, C_final*
template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void 
innerSpGEMM_nohash(const CSR<IT,NT> & A, const CSC<IT,NT> & B, CSR<IT,NT> & C_final, const CSR<IT,NT> & M, MultiplyOperation multop, AddOperation addop, unsigned threadCount)
{
    CSR<IT,NT> C;
    
    //*A^2*
    C.rows = M.rows;
    C.cols = M.cols; // B ?=A
    C.nnz = M.nnz;
    C.zerobased = true;

    C.rowptr = my_malloc<IT>(M.rows + 1);
    C.colids = my_malloc<IT>(M.nnz);
    C.values = my_malloc<NT>(M.nnz);

    for (IT i = 0; i < C.rows; ++i)
        C.rowptr[i] = M.rowptr[i];
    
    for (IT i = 0; i < C.nnz; ++i) {
        C.colids[i] = M.colids[i]; // unnecessary
        C.values[i] = 0;
    }
    
    BIN<IT, NT> bin(A.rows, IMB_PWMIN, threadCount);

    /* Set max bin */
    // Double check, changed 3rd param to colptr
    bin.set_max_bin(A.rowptr, A.colids, B.colptr, C.rows, C.cols);

    IT numThreads;
    #pragma omp parallel num_threads(threadCount)
    {
        numThreads = omp_get_num_threads();
    }

    vector<IT> th_nnz(numThreads, 0);
    vector<IT> rownnz(C.rows, 0);

    IT rowPerThread = (M.rows + numThreads -1) / numThreads;
    #pragma omp parallel  num_threads(threadCount)
    {
        IT i, start_row, end_row, col;
        IT tid;
        
        tid = omp_get_thread_num();
        // start_row  = bin.rows_offset[tid];
        // end_row = bin.rows_offset[tid + 1];
        start_row = rowPerThread * tid;
        end_row = min(rowPerThread * (tid+1), M.rows);

        // each th keeps track of active nnz in C (not all from Mask)   
        //* blocks of rows the mask *
        for (i = start_row; i < end_row; ++i) {
            IT j, cur_col, nnz_r, nnz_c;
            IT cur_row = i;
            NT t_val = 0; 
            bool active = false;
       
            //* nonzeros of the row over the mask *  
            for (j = M.rowptr[i]; j < M.rowptr[i + 1]; ++j) {
           
                cur_col = M.colids[j];      
                nnz_r = A.rowptr[cur_row]; 
                nnz_c = B.colptr[cur_col]; 
                t_val = 0;
                active = false;

                //*dot product between row of A and col of B 
                while(nnz_r < A.rowptr[cur_row+1] && nnz_c < B.colptr[cur_col+1]){

                    if(A.colids[nnz_r] < B.rowids[nnz_c])
                        nnz_r++;
                    else if(A.colids[nnz_r] > B.rowids[nnz_c])
                        nnz_c++;
                    else { //A.colids[nnz_r] == B.rowids[nnz_c];
                        t_val = addop(t_val, multop(A.values[nnz_r], B.values[nnz_c]));
                        nnz_r++, 
                        nnz_c++;
                        active = true;
                    }
                }
                if(active) {// active nnz, shrink output accordingly
                    IT loc = M.rowptr[start_row] + th_nnz[tid];
                    C.colids[loc] = M.colids[j];
                    C.values[loc] = t_val;
                    th_nnz[tid]++;
                    rownnz[i]++;
                }  
            }          
        }
        #pragma omp barrier
    }

    //shrink C

    //* sequentially create global rowptr for final shrinked C*
    for (IT i = 0; i < C.rows; ++i)
         C_final.nnz += rownnz[i];
    
    C_final.rows = C.rows;
    C_final.cols = C.cols;
    C_final.zerobased = true;

    C_final.rowptr = my_malloc<IT>(C.rows + 1);
    C_final.colids = my_malloc<IT>(C_final.nnz);
    C_final.values = my_malloc<NT>(C_final.nnz);

    memcpy (C_final.colids, C.colids, th_nnz[0] * sizeof(IT)) ;
    memcpy (C_final.values, C.values, th_nnz[0] * sizeof(NT)) ;
         
    IT dest = 0;
    for (IT i = 1; i < numThreads; ++i) {
        IT loc = min(i * rowPerThread, A.rows);
        dest += th_nnz[i-1];
        memcpy (C_final.colids + dest, C.colids + A.rowptr[loc], th_nnz[i] * sizeof(C.colids[0]));
        memcpy (C_final.values + dest, C.values + A.rowptr[loc], th_nnz[i] * sizeof(C.values[0])); 
    }

    //TODO:: optimize prefix sum
    C_final.rowptr[0] = 0;
    for (IT i = 1; i <= C_final.rows; ++i) {
        C_final.rowptr[i] =  C_final.rowptr[i-1] + rownnz[i-1];//A.rowptr[rowPerThread * i];
    }

    // cout << "Dot SpGEMM with Mask C_final" << endl;
    // for (int i = 0; i < 10; ++i){
    //     cout << i << " : " << C_final.rowptr[i] << " ";
    //     for (int j = C_final.rowptr[i];  j < C_final.rowptr[i+1]; ++j)
    //         cout << C_final.colids[j] << " " << C_final.values[j] << ", ";
    //     cout << endl;
    // }
    // cout << endl;
    C.make_empty();
}


