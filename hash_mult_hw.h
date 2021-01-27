#ifndef _HASH_MULT_HW_
#define _HASH_MULT_HW_


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// #include <immintrin.h>
//#include <zmmintrin.h>
#include <algorithm>

#include "utility.h"
#include "CSR.h"
#include "BIN.h"

/* SpGEMM Specific Parameters */
#define HASH_SCAL 107 // Set disjoint number to SH_SIZE
#define IMB_PWMIN 8
#define B_PWMIN 8
#define VEC_LENGTH 8
#define VEC_LENGTH_BIT 3
#define VEC_LENGTH_LONG 4
#define VEC_LENGTH_LONG_BIT 2


template <unsigned threadCount, class IT, class NT>
inline void hash_symbolic_kernel(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, BIN<IT, NT> &bin)
{
#pragma omp parallel num_threads(threadCount)
    {
        IT i, tid, start_row, end_row;
        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        IT *check = bin.local_hash_table_id[tid];

        for (i = start_row; i < end_row; ++i) {
            IT j, k, bid;
            IT key, hash, old;
            IT nz, SH_ROW;
            IT t_acol;

            nz = 0;
            bid = bin.bin_id[i];

            if (bid > 0) {
                SH_ROW = IMB_PWMIN << (bid - 1);
                for (j = 0; j < SH_ROW; ++j) {
                    check[j] = -1;
                }

                for (j = arpt[i]; j < arpt[i + 1]; ++j) {
                    t_acol = acol[j];
                    for (k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (SH_ROW - 1);
                        while (1) {
                            if (check[hash] == key) {
                                break;
                            }
                            else if (check[hash] == -1) {
                                check[hash] = key;
                                nz++;
                                break;
                            }
                            else {
                                hash = (hash + 1) & (SH_ROW - 1); //hash = (hash + 1) % SH_ROW
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}


template <unsigned threadCount, class IT, class NT>
inline void hash_symbolic(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, IT *crpt, BIN<IT, NT> &bin, const IT nrow, IT *nnz)
{
    IT i;
    hash_symbolic_kernel<threadCount>(arpt, acol, brpt, bcol, bin);

    /* Set row pointer of matrix C */
    scan(bin.row_nz, crpt, nrow + 1);
    *nnz = crpt[nrow];
}

template <typename IT, typename NT>
bool sort_less(const pair<IT, NT> &left,const pair<IT, NT> &right)
{
    return left.first < right.first;
}

template <bool sortOutput, unsigned threadCount, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric(const IT *arpt, const IT *acol, const NT *aval, const IT *brpt, const IT *bcol, const NT *bval, const IT *crpt, IT *ccol, NT *cval,const BIN<IT, NT> &bin, const MultiplyOperation multop, const AddOperation addop)
{
#pragma omp parallel num_threads(threadCount)
    {
        IT i, tid, start_row, end_row;
        IT *shared_check;
        NT *shared_value;

        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        shared_check = bin.local_hash_table_id[tid];
        shared_value = bin.local_hash_table_val[tid];

        for (i = start_row; i < end_row; ++i) {
            IT j, k, bid, index;
            IT SH_ROW;
            IT t_acol, hash, key, offset;
            NT t_aval, t_val;

            bid = bin.bin_id[i];

            if (bid > 0) {

                offset = crpt[i];
                SH_ROW = B_PWMIN << (bid - 1);

                for (j = 0; j < SH_ROW; ++j) {
                    shared_check[j] = -1;
                }

                for (j = arpt[i]; j < arpt[i + 1]; ++j) {
                    t_acol = acol[j];
                    t_aval = aval[j];
                    for (k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        t_val = multop(t_aval, bval[k]);

                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (SH_ROW - 1);
                        while (1) {
                            if (shared_check[hash] == key) {
                                shared_value[hash] = addop(t_val, shared_value[hash]);
                                break;
                            }
                            else if (shared_check[hash] == -1) {
                                shared_check[hash] = key;
                                shared_value[hash] = t_val;
                                break;
                            }
                            else {
                                hash = (hash + 1) & (SH_ROW - 1);
                            }
                        }
                    }
                }
                index = 0;
                if (sortOutput) {
                    IT nz = crpt[i + 1] - offset;
                    vector<pair<IT, NT>> p_vec(nz);
                    for (j = 0; j < SH_ROW; ++j) {
                        if (shared_check[j] != -1) {
                            p_vec[index++] = make_pair(shared_check[j], shared_value[j]);
                        }
                    }
                    sort(p_vec.begin(), p_vec.end(), sort_less<IT, NT>);
                    for (j = 0; j < index; ++j) {
                        ccol[offset + j] = p_vec[j].first;
                        cval[offset + j] = p_vec[j].second;
                    }
                }
                else {
                    for (j = 0; j < SH_ROW; ++j) {
                        if (shared_check[j] != -1) {
                            ccol[offset + index] = shared_check[j];
                            cval[offset + index] = shared_value[j];
                            index++;
                        }
                    }
                }
            }
        }
    }
}

template <unsigned threadCount, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    BIN<IT, NT> bin(a.rows, IMB_PWMIN, threadCount);

    c.rows = a.rows;
    c.cols = b.cols;
    c.zerobased = true;

    /* Set max bin */
    bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);

    /* Create hash table (thread local) */
    bin.create_local_hash_table(c.cols);

    /* Symbolic Phase */
    c.rowptr = my_malloc<IT>(c.rows + 1);
    hash_symbolic<threadCount>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr, bin, c.rows, &(c.nnz));

    c.colids = my_malloc<IT>(c.nnz);
    c.values = my_malloc<NT>(c.nnz);

    // only non-vector case implemented
    hash_numeric<sortOutput, threadCount>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
}

template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    HashSpGEMM<false, sortOutput, IT, NT, MultiplyOperation, AddOperation>(a, b, c, multop, addop);
}

template <typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    HashSpGEMM<false, true, IT, NT, MultiplyOperation, AddOperation>(a, b, c, multop, addop);
}

#endif
