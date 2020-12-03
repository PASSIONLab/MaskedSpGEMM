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

template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void 
innerSpGEMM_nohash(const CSR<IT,NT> & A, const CSC<IT,NT> & B, CSR<IT,NT> & C, MultiplyOperation multop, AddOperation addop)
{
    C.rows = A.rows;
    C.cols = B.cols;
    C.nnz = A.nnz;
    C.zerobased = true;

    C.rowptr = my_malloc<IT>(A.rows + 1);
    C.colids = my_malloc<IT>(A.nnz);
    C.values = my_malloc<NT>(A.nnz);

    for (int i = 0; i < C.rows; ++i) 
        C.rowptr[i] = A.rowptr[i];
    
    for (int i = 0; i < A.nnz; ++i) {
        C.colids[i] = A.colids[i];
        C.values[i] = 0;
    }

    int numThreads; 
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }

    IT rowPerThread = (A.rows + numThreads -1) / numThreads;
    #pragma omp parallel
    {
        int i, tid, start_row, end_row, col;
        IT *shared_check;
        NT *shared_value;

        tid = omp_get_thread_num();
        start_row = rowPerThread * tid;
        end_row = min(rowPerThread * (tid+1), A.rows);

        //*blocks of rows the mask*
        for (i = start_row; i < end_row; ++i) {
            int j, cur_col, nnz_r, nnz_c;
            int cur_row = i; 
            NT t_val = 0; 
       
            //*nonzeros of the row over the mask*  
            for (j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
           
                cur_col = A.colids[j];      
                nnz_r = A.rowptr[cur_row]; 
                nnz_c = B.colptr[cur_col]; 
                t_val = 0;

                //*dot product between row of A and col of B 
                while(nnz_r < A.rowptr[cur_row+1] && nnz_c < B.colptr[cur_col+1]){

                    if(A.colids[nnz_r] < B.rowids[nnz_c])
                        nnz_r++;
                    else if(A.colids[nnz_r] > B.rowids[nnz_c])
                        nnz_c++;
                    else { //A.colids[nnz_r] == B.rowids[nnz_c];
                        t_val += multop(A.values[nnz_r], B.values[nnz_c]);
                        nnz_r++, 
                        nnz_c++;
                    }
                }
                C.values[j] = t_val;  
            }          
        }
    }
    //shrink C
}

// template <typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
// void 
// innerSpGEMM(const CSR<IT,NT> & A, const CSC<IT,NT> & B, CSR<IT,NT> & C, MultiplyOperation multop, AddOperation addop)
// {
//     BIN<IT, NT> bin(A.rows, IMB_PWMIN);
//     C.rows = A.rows;
//     C.cols = B.cols;
//     C.zerobased = true;

//     /* Set max bin */
//     bin.set_max_bin(A.rowptr, A.colids, B.rowptr, C.rows, C.cols);

//     /* Create hash table (thread local) */
//     bin.create_local_hash_table(C.cols);

//     /* Symbolic Phase */
//     C.rowptr = my_malloc<IT>(C.rows + 1);
//     // hash_numeric<sortOutput>( c.rowptr, c.colids, c.values, bin, multop, addop);
//     // inline void hash_numeric(  const IT *crpt, IT *ccol, NT *cval,const BIN<IT, NT> &bin, const MultiplyOperation multop, const AddOperation addop)
// do symbolic
//     C.colids = my_malloc<IT>(C.nnz);
//     C.values = my_malloc<NT>(C.nnz);

//     IT *arpt = A.rowptr;
//     IT *acol = A.colids;
//     NT*aval = A.values;
//     IT *bcpt = B.colptr;
//     IT *brow = B.rowids;
//     NT*bval = B.values;
//     IT *crpt = C.rowptr;
//     IT *ccol = C.colids;
//     NT*cval = C.values;

//     #pragma omp parallel
//     {
//         int i, tid, start_row, end_row;
//         IT *shared_check;
//         NT *shared_value;

//         tid = omp_get_thread_num();
//         start_row = bin.rows_offset[tid];
//         end_row = bin.rows_offset[tid + 1];

//         // shared_check = bin.local_hash_table_id[tid];
//         // shared_value = bin.local_hash_table_val[tid];

//         for (i = start_row; i < end_row; ++i) {
//             int j, k, bid, index;
//             int SH_ROW;
//             int t_acol, hash, key, offset;
//             NT t_aval, t_val;

//             bid = bin.bin_id[i];

//             if (bid > 0) {

//                 offset = crpt[i];
//                 SH_ROW = B_PWMIN << (bid - 1);

//                 for (j = 0; j < SH_ROW; ++j) {
//                     shared_check[j] = -1;
//                 }

//                 for (j = arpt[i]; j < arpt[i + 1]; ++j) {
//                     t_acol = acol[j];
//                     t_aval = aval[j];
//                     // key = acol[k];
                    
//                     for (k = bcpt[t_acol]; k < bcpt[t_acol + 1]; ++k) {
//                         t_val = multop(t_aval, bval[k]);

              
//                         // hash = (key * HASH_SCAL) & (SH_ROW - 1);
//                         // while (1) {
//                         //     if (shared_check[hash] == key) {
//                         //         shared_value[hash] = addop(t_val, shared_value[hash]);
//                         //         break;
//                         //     }
//                         //     else if (shared_check[hash] == -1) {
//                         //         shared_check[hash] = key;
//                         //         shared_value[hash] = t_val;
//                         //         break;
//                         //     }
//                         //     else {
//                         //         hash = (hash + 1) & (SH_ROW - 1);
//                         //     }
//                         // }
//                     }
//                 }
//                 index = 0;
            
//                 for (j = 0; j < SH_ROW; ++j) {
//                     if (shared_check[j] != -1) {
//                         ccol[offset + index] = shared_check[j];
//                         cval[offset + index] = shared_value[j];
//                         index++;
//                     }
//                 }
                
//             }
//         }
//     }

// }

