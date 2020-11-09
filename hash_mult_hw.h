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

//#define VECTORIZE

template <class IT, class NT>
inline void hash_symbolic_kernel(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, BIN<IT, NT> &bin)
{
#pragma omp parallel
    {
        int i, tid, start_row, end_row;
        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        IT *check = bin.local_hash_table_id[tid];

        for (i = start_row; i < end_row; ++i) {
            int j, k, bid;
            int key, hash, old;
            int nz, SH_ROW;
            int t_acol;

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

template <class NT>
inline void hash_symbolic_vec_kernel(const int *arpt, const int *acol, const int *brpt, const int *bcol, BIN<int, NT> &bin)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi32(-1);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
#endif

#pragma omp parallel
    {
        int i, tid, start_row, end_row;
        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        int *check = bin.local_hash_table_id[tid];

        for (i = start_row; i < end_row; ++i) {
            int j, k, bid;
            int key, hash, old;
            int nz, SH_ROW, table_size;
            int t_acol;
#ifdef VECTORIZE
            __m256i key_m, check_m;
            __m256i mask_m;
            int mask;
#endif
            nz = 0;
            bid = bin.bin_id[i];

            if (bid > 0) {
                table_size = IMB_PWMIN << (bid - 1);
                SH_ROW = table_size >> VEC_LENGTH_BIT;
                for (j = 0; j < table_size; ++j) {
                    check[j] = -1;
                }

                for (j = arpt[i]; j < arpt[i + 1]; ++j) {
                    t_acol = acol[j];
                    for (k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (SH_ROW - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi32(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi32(check + (hash << VEC_LENGTH_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi32(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma simd
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (check[(hash << VEC_LENGTH_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi32(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 2;
#else
                                cur_nz = VEC_LENGTH;
#pragma simd
                                for (int l = VEC_LENGTH - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) {
                                    check[(hash << VEC_LENGTH_BIT) + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (SH_ROW - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

template <class NT>
inline void hash_symbolic_vec_kernel(const long long int *arpt, const long long int *acol, const long long int *brpt, const long long int *bcol, BIN<long long int, NT> &bin)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi64x(-1);
    const __m256i true_m = _mm256_set1_epi64x(0xffffffffffffffff);
#endif

#pragma omp parallel
    {
        long long int i, tid, start_row, end_row;
        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        long long int *check = bin.local_hash_table_id[tid];

        for (i = start_row; i < end_row; ++i) {
            long long int j, k, bid;
            long long int key, hash, old;
            long long int nz, SH_ROW, table_size;
            long long int t_acol;
#ifdef VECTORIZE
            __m256i key_m, check_m;
            __m256i mask_m;
            int mask;
#endif
            nz = 0;
            bid = bin.bin_id[i];

            if (bid > 0) {
                table_size = IMB_PWMIN << (bid - 1);
                SH_ROW = table_size >> VEC_LENGTH_LONG_BIT;
                for (j = 0; j < table_size; ++j) {
                    check[j] = -1;
                }

                for (j = arpt[i]; j < arpt[i + 1]; ++j) {
                    t_acol = acol[j];
                    for (k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (SH_ROW - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi64x(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi64(check + (hash << VEC_LENGTH_LONG_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi64(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma simd
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                long long int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi64(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 3;
#else
                                cur_nz = VEC_LENGTH_LONG;
#pragma simd
                                for (int l = VEC_LENGTH_LONG - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) {
                                    check[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (SH_ROW - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

template <bool vectorProbing, class IT, class NT>
inline void hash_symbolic(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, IT *crpt, BIN<IT, NT> &bin, const IT nrow, IT *nnz)
{
    IT i;
    if (vectorProbing) {
        hash_symbolic_vec_kernel(arpt, acol, brpt, bcol, bin);
    }
    else {
        hash_symbolic_kernel(arpt, acol, brpt, bcol, bin);
    }

    /* Set row pointer of matrix C */
    scan(bin.row_nz, crpt, nrow + 1);
    *nnz = crpt[nrow];
}

template <typename IT, typename NT>
bool sort_less(const pair<IT, NT> &left,const pair<IT, NT> &right)
{
    return left.first < right.first;
}

template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric(const IT *arpt, const IT *acol, const NT *aval, const IT *brpt, const IT *bcol, const NT *bval, const IT *crpt, IT *ccol, NT *cval,const BIN<IT, NT> &bin, const MultiplyOperation multop, const AddOperation addop)
{
#pragma omp parallel
    {
        int i, tid, start_row, end_row;
        IT *shared_check;
        NT *shared_value;

        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        shared_check = bin.local_hash_table_id[tid];
        shared_value = bin.local_hash_table_val[tid];

        for (i = start_row; i < end_row; ++i) {
            int j, k, bid, index;
            int SH_ROW;
            int t_acol, hash, key, offset;
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
                    int nz = crpt[i + 1] - offset;
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

template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const int *arpt, const int *acol, const NT *aval, const int *brpt, const int *bcol, const NT *bval, const int *crpt, int *ccol, NT *cval, const BIN<int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi32(-1);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
#endif

#pragma omp parallel
    {
        int max_table_size = 0;

        int i, tid, start_row, end_row;
        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        int *shared_check = bin.local_hash_table_id[tid];
        NT *shared_value = bin.local_hash_table_val[tid];

        for (i = start_row; i < end_row; ++i) {
            int j, k, bid;
            int SH_ROW, table_size;
            int t_acol, hash, key, offset, index;
            NT t_aval, t_val;
#ifdef VECTORIZE
            __m256i key_m, check_m, mask_m;
            int mask;
#endif
            bid = bin.bin_id[i];

            if (bid > 0) {
                offset = crpt[i];
                table_size = B_PWMIN << (bid - 1);
                SH_ROW = table_size >> VEC_LENGTH_BIT;

                for (j = 0; j < table_size; ++j) {
                    shared_check[j] = -1;
                }

                for (j = arpt[i]; j < arpt[i + 1]; ++j) {
                    t_acol = acol[j];
                    t_aval = aval[j];
                    for (k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        t_val = multop(t_aval, bval[k]);

                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (SH_ROW - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi32(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi32(shared_check + (hash << VEC_LENGTH_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi32(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                int target = __builtin_ctz(mask) >> 2;
                                shared_value[(hash << VEC_LENGTH_BIT) + target] += t_val;
                                break;
                            }
#else
                            int flag = -1;
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (shared_check[(hash << VEC_LENGTH_BIT) + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                shared_value[(hash << VEC_LENGTH_BIT) + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi32(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 2;
#else
                                cur_nz = VEC_LENGTH;
                                for (int l = 0; l < VEC_LENGTH; ++l) {
                                    if (shared_check[(hash << VEC_LENGTH_BIT) + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) {
                                    shared_check[(hash << VEC_LENGTH_BIT) + cur_nz] = key;
                                    shared_value[(hash << VEC_LENGTH_BIT) + cur_nz] = t_val;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (SH_ROW - 1);
                                }
                            }
                        }
                    }
                }

                index = 0;
                if (sortOutput) {
                    int nz = crpt[i + 1] - offset;
                    vector<pair<int, NT>> p_vec(nz);
                    for (j = 0; j < table_size; ++j) {
                        if (shared_check[j] != -1) {
                            p_vec[index++] = make_pair(shared_check[j], shared_value[j]);
                        }
                    }
                    sort(p_vec.begin(), p_vec.end(), sort_less<int, NT>);
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

template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const long long int *arpt, const long long int *acol, const NT *aval, const long long int *brpt, const long long int *bcol, const NT *bval, const long long int *crpt, long long int *ccol, NT *cval, const BIN<long long int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi64x(-1);
    const __m256i true_m = _mm256_set1_epi64x(0xffffffffffffffff);
#endif

#pragma omp parallel
    {
        long long int max_table_size = 0;

        long long int i, tid, start_row, end_row;
        tid = omp_get_thread_num();
        start_row = bin.rows_offset[tid];
        end_row = bin.rows_offset[tid + 1];

        long long int *shared_check = bin.local_hash_table_id[tid];
        NT *shared_value = bin.local_hash_table_val[tid];

        for (i = start_row; i < end_row; ++i) {
            long long int j, k, bid;
            long long int SH_ROW, table_size;
            long long int t_acol, hash, key, offset, index;
            NT t_aval, t_val;
#ifdef VECTORIZE
            __m256i key_m, check_m, mask_m;
            int mask;
#endif
            bid = bin.bin_id[i];

            if (bid > 0) {
                offset = crpt[i];
                table_size = B_PWMIN << (bid - 1);
                SH_ROW = table_size >> VEC_LENGTH_LONG_BIT;

                for (j = 0; j < table_size; ++j) {
                    shared_check[j] = -1;
                }

                for (j = arpt[i]; j < arpt[i + 1]; ++j) {
                    t_acol = acol[j];
                    t_aval = aval[j];
                    for (k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        t_val = multop(t_aval, bval[k]);

                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (SH_ROW - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi64x(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi64(shared_check + (hash << VEC_LENGTH_LONG_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi64(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                int target = __builtin_ctz(mask) >> 3;
                                shared_value[(hash << VEC_LENGTH_LONG_BIT) + target] += t_val;
                                break;
                            }
#else
                            int flag = -1;
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (shared_check[(hash << VEC_LENGTH_LONG_BIT) + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                shared_value[(hash << VEC_LENGTH_LONG_BIT) + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi64(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 3;
#else
                                cur_nz = VEC_LENGTH_LONG;
                                for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                    if (shared_check[(hash << VEC_LENGTH_LONG_BIT) + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) {
                                    shared_check[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = key;
                                    shared_value[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = t_val;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (SH_ROW - 1);
                                }
                            }
                        }
                    }
                }

                index = 0;
                if (sortOutput) {
                    int nz = crpt[i + 1] - offset;
                    vector<pair<long long int, NT>> p_vec(nz);
                    for (j = 0; j < table_size; ++j) {
                        if (shared_check[j] != -1) {
                            p_vec[index++] = make_pair(shared_check[j], shared_value[j]);
                        }
                    }
                    sort(p_vec.begin(), p_vec.end(), sort_less<long long int, NT>);
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

template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    BIN<IT, NT> bin(a.rows, IMB_PWMIN);

    c.rows = a.rows;
    c.cols = b.cols;
    c.zerobased = true;

    /* Set max bin */
    bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);

    /* Create hash table (thread local) */
    bin.create_local_hash_table(c.cols);

    /* Symbolic Phase */
    c.rowptr = my_malloc<IT>(c.rows + 1);
    hash_symbolic<vectorProbing>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr, bin, c.rows, &(c.nnz));

    c.colids = my_malloc<IT>(c.nnz);
    c.values = my_malloc<NT>(c.nnz);

    /* Numeric Phase */
    if (vectorProbing) {
        hash_numeric_vec<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
    }
    else {
        hash_numeric<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
    }
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

