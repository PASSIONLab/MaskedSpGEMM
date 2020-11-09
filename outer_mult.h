#include "CSC.h"
#include "CSR.h"
#include "Triple.h"
#include "radix_sort/radix_sort.hpp"
#include "utility.h"
#include <map>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <cstring>
#include <numa.h>
#include <thread>
#include <cstdlib>
#include <chrono>
#include<set>

using namespace std;

static uint32_t nrows_of_A;
static uint32_t nrows_per_blocker;
// static int *rows_to_blockers;
// static int *flops_by_row_blockers;
#define TBB true
#define SIZE 16

template <typename IT>
uint16_t fast_mod(const IT input, const int ceil) {
    return input >= ceil ? input % ceil : input;
}


template <typename IT, typename NT>
uint64_t getFlop(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    uint64_t flop = 0;

#pragma omp parallel for reduction(+ : flop)
    for (IT i = 0; i < A.cols; ++i)
    {
        IT colnnz = A.colptr[i + 1] - A.colptr[i];
        IT rownnz = B.rowptr[i + 1] - B.rowptr[i];
        flop += (colnnz * rownnz);
    }
    return flop;
}

template <typename IT, typename NT>
void do_static_symbolic(const CSC<IT, NT>& A, const CSR<IT, NT>& B, IT startIdx, IT endIdx,
                 uint32_t nrows_per_blocker, uint16_t num_blockers,
                 IT *flops_by_row_blockers, IT& total_flops)
{
    #pragma omp parallel for reduction(+ : flops_by_row_blockers[:num_blockers])
    for (IT i = startIdx; i < endIdx; ++i)
    {
        IT rownnz = B.rowptr[i + 1] - B.rowptr[i];
        for (IT j = A.colptr[i]; j < A.colptr[i + 1]; ++j)
        {
            uint16_t row_blocker_id = A.rowids[j] / nrows_per_blocker;
            // cout << "row_blocker_id " << row_blocker_id << endl;
            flops_by_row_blockers[row_blocker_id] += rownnz;
        }
    }
    for (IT i = 0; i < num_blockers; ++i)
    {
        total_flops += flops_by_row_blockers[i];
    }
}

template <typename IT, typename NT>
void do_symbolic(const CSC<IT, NT>& A, const CSR<IT, NT>& B, IT startIdx, IT endIdx,
                 uint16_t num_blockers, IT* flops_by_rows, IT* rows_to_blockers,
                 IT* flops_by_row_blockers, IT& total_flops)
{
    double avg_volumn = 0.0;
    double cur_volumn = 0.0;
    uint16_t cur_blocker_id = 0;

// #pragma omp parallel
{
    // #pragma omp for reduction(+ : flops_by_rows[:A.rows])
    for (IT i = startIdx; i < endIdx; ++i)
    {
        IT rownnz = B.rowptr[i + 1] - B.rowptr[i];
        for (IT j = A.colptr[i]; j < A.colptr[i + 1]; ++j)
        {
            flops_by_rows[A.rowids[j]] += rownnz;
        }
    }
    #pragma omp parallel for reduction(+ : total_flops)
    for (IT i = 0; i<A.rows; ++i)
        total_flops += flops_by_rows[i];
}
    avg_volumn = total_flops / num_blockers;
    rows_to_blockers[0] = 0;
    for (IT i=0; i<A.rows; ++i)
    {
        cur_volumn += flops_by_rows[i];
        flops_by_row_blockers[cur_blocker_id] = cur_volumn;
        rows_to_blockers[i] = cur_blocker_id;
        if (cur_volumn > avg_volumn)
        {
            cur_blocker_id ++;
            cur_volumn = 0;
        }
    }
    // for (uint16_t i = 0 ; i< num_blockers; ++i)
    //     cout << "BlockerId = " << i << " RowId = " << rows_to_blockers[i] << endl;
}

template <typename IT, typename NT>
int64_t getReqMemory(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    uint64_t flop = getFlop(A, B);
    return flop * sizeof(int64_t);
}

struct ExtractKey
{
    inline int64_t operator()(tuple<int32_t, int32_t, double> tup)
    {
        int64_t res = std::get<0>(tup);
        res         = (res << 32);
        res         = res | (int64_t)(uint32_t) std::get<1>(tup);
        return res;
    }
};


struct ExtractKey2
{
    inline uint32_t operator()(tuple<int32_t, int32_t, double> tup)
    {
        // 32768 for S23
        // 256 for S16
		return ((std::get<0>(tup) % nrows_per_blocker) << 20 | (uint32_t) std::get<1>(tup));
        // return (std::get<0>(tup) << 16) | ((uint32_t) std::get<1>(tup));
        // return ((std::get<0>(tup) % flops_by_row_blockers[rows_to_blockers[std::get<0>(tup)]] << 16) | ((uint32_t) std::get<1>(tup)));
        // return (((rows_to_blockers[std::get<0>(tup)] % (flops_by_row_blockers[std::get<0>(tup)])) << 20) | ((uint32_t) std::get<1>(tup)));
    }
};

template <typename IT, typename NT>
bool compareTuple (tuple<IT, IT, NT> t1, tuple<IT, IT, NT> t2)
{
    if (std::get<0>(t1) < std::get<0>(t2))
        return true;
    else if (std::get<0>(t1) == std::get<0>(t2) && std::get<1>(t1) < std::get<1>(t2))
        return true;
    return false;
    // if (std::get<1>(t1) != std::get<1>(t2))
    //     return false;
    // if (std::get<0>(t1) != std::get<0>(t2))
    //     return false;
    // return true;
}

template <typename IT, typename NT>
inline bool isTupleEqual (tuple<IT, IT, NT> t1, tuple<IT, IT, NT> t2)
{
    if (std::get<1>(t1) != std::get<1>(t2))
        return false;
    if (std::get<0>(t1) != std::get<0>(t2))
        return false;
    return true;
}

template <typename IT, typename NT>
inline void doRadixSort(tuple<IT, IT, NT>* begin, tuple<IT, IT, NT>* end, tuple<IT, IT, NT>* buffer)
{
    radix_sort(begin, end, buffer, ExtractKey2());
    // sort(begin, end, compareTuple<IT, NT>);
}

template <typename IT, typename NT>
inline IT doMerge(tuple<IT, IT, NT>* vec, IT length)
{
    if (length == 0) return 0;
    ExtractKey op = ExtractKey();
    IT i          = 0;
    IT j          = 1;

    while (i < length && j < length)
    {
        if (j < length && isTupleEqual (vec[i], vec[j]))
            std::get<2>(vec[i]) += std::get<2>(vec[j]);
        else
        {
            // vec[++i] = std::move(vec[j]);
            ++i;
            std::get<0>(vec[i]) = std::get<0>(vec[j]);
            std::get<1>(vec[i]) = std::get<1>(vec[j]);
            std::get<2>(vec[i]) = std::get<2>(vec[j]);
        }
        ++j;
    }
    return i + 1;
}

template <typename IT>
void initializeBlockerBoundary(IT* nums_per_col_blocker, uint16_t num_blockers, IT* blocker_begin_ptr,
                               IT* blocker_end_ptr)
{
    blocker_begin_ptr[0] = 0;
    blocker_end_ptr[0]   = 0;
    for (uint16_t blocker_index = 1; blocker_index < num_blockers; ++blocker_index)
    {
        blocker_begin_ptr[blocker_index] = blocker_begin_ptr[blocker_index - 1] + nums_per_col_blocker[blocker_index - 1];
        blocker_end_ptr[blocker_index] = blocker_begin_ptr[blocker_index];
    }
}

template <typename IT, typename NT>
void OuterSpGEMM_stage(const CSC<IT, NT>& A, const CSR<IT, NT>& B, IT startIdx, IT endIdx, CSR<IT, NT>& C, \
    int nblockers, int nblockchars, int partition)
{
    typedef tuple<IT, IT, NT> TripleNode;
    const uint16_t nthreads = omp_get_max_threads();
    uint16_t num_blockers = nblockers;
    const uint16_t block_width = nblockchars;
    nrows_of_A = A.rows;
	nrows_per_blocker = A.rows <= num_blockers * 64 ? 64 : (A.rows + num_blockers - 1) / num_blockers;

    IT total_flop = 0;

    IT *flops_by_row_blockers = my_malloc<IT>(num_blockers);
    IT *flops_by_rows = my_malloc<IT>(A.rows);
    IT *nnz_by_row = my_malloc<IT>(A.rows);

    IT *global_blocker_counters = my_malloc<IT>(num_blockers);
    TripleNode **global_blockers = my_malloc<TripleNode*>(num_blockers);
    IT **local_blocker_counters = my_malloc<IT*>(nthreads);
    TripleNode **local_blockers = my_malloc<TripleNode*>(nthreads);
    TripleNode **sorting_buffer = my_malloc<TripleNode*>(nthreads);
    IT *nnz_per_row_blocker = my_malloc<IT>(num_blockers);
	do_static_symbolic(A, B, 0, A.cols, nrows_per_blocker, num_blockers, flops_by_row_blockers, total_flop);
    for (uint16_t blocker_id=0; blocker_id<num_blockers; ++blocker_id)
        global_blockers[blocker_id] = static_cast<TripleNode*>(::operator new(SIZE * flops_by_row_blockers[blocker_id]));
    IT max_flops_in_row_blockers = *std::max_element(flops_by_row_blockers, flops_by_row_blockers + num_blockers);
    // IT min_flops_in_row_blockers = *std::min_element(flops_by_row_blockers, flops_by_row_blockers + num_blockers);
    // uint64_t avg_flops_in_row_blockers = 0;

    // for (IT i = 0; i < num_blockers; ++i) {
    //     avg_flops_in_row_blockers += flops_by_row_blockers[i];
    //     max_flops_in_row_blockers = max(max_flops_in_row_blockers, flops_by_row_blockers[i]);
    // }
    // cout << "avg_flops_in_row_blockers = " << avg_flops_in_row_blockers / num_blockers << "   max_flops_in_row_blockers = " << max_flops_in_row_blockers << "    min_flops_in_row_blockers = " << min_flops_in_row_blockers << endl;
#pragma omp parallel
    {
        uint16_t thread_id = omp_get_thread_num();
        TripleNode *begin_local_blockers, *cur_local_blockers, *end_local_blockers, *cur_global_blockers;
        local_blockers[thread_id] = static_cast<TripleNode*>(::operator new(SIZE * num_blockers * block_width));
        local_blocker_counters[thread_id] = my_malloc<IT>(num_blockers);
        sorting_buffer[thread_id] = static_cast<TripleNode*>(::operator new(SIZE * max_flops_in_row_blockers));
// computing phase
#pragma omp for nowait
        for (IT idx = startIdx; idx < endIdx; ++idx)
        {
            // std::set<int>::iterator it = dense_cols.find(idx);
            // if (it != dense_cols.end()) continue;
            for (IT j = A.colptr[idx]; j < A.colptr[idx + 1]; ++j) // ncols(A) * 4
            {
                IT rowid = A.rowids[j]; // nnz(A) * 4
                uint16_t row_blocker_id = rowid / nrows_per_blocker;
                begin_local_blockers = local_blockers[thread_id] + row_blocker_id * block_width;
                cur_local_blockers = begin_local_blockers + local_blocker_counters[thread_id][row_blocker_id];
                end_local_blockers = begin_local_blockers + block_width;
                for (IT k = B.rowptr[idx]; k < B.rowptr[idx + 1]; ++k)   // nrows(B) * 4
                {
                    // *cur_local_blockers = std::move(TripleNode(A.rowids[j], B.colids[k], A.values[j] * B.values[k]));
                    std::get<0>(*cur_local_blockers) = A.rowids[j];
                    std::get<1>(*cur_local_blockers) = B.colids[k];
                    std::get<2>(*cur_local_blockers) = A.values[j] * B.values[k];
					cur_local_blockers++;
                    if (cur_local_blockers == end_local_blockers) // flop * 16
                    {
                        // cur_global_blockers = global_blockers[row_blocker_id] + __sync_fetch_and_add(&global_blocker_counters[row_blocker_id], block_width);
                        // for (IT offset=0; offset<block_width; ++offset)
                        // {
                        //     std::get<0>(cur_global_blockers[offset]) = std::get<0>(begin_local_blockers[offset]);
                        //     std::get<1>(cur_global_blockers[offset]) = std::get<1>(begin_local_blockers[offset]);
                        //     std::get<2>(cur_global_blockers[offset]) = std::get<2>(begin_local_blockers[offset]);
                        //     // cur_global_blockers[offset] = begin_local_blockers[offset];
                        // }
                        std::memcpy(
                            global_blockers[row_blocker_id] + __sync_fetch_and_add(&global_blocker_counters[row_blocker_id], block_width),
                            begin_local_blockers,
                            block_width * SIZE
                        );
                        cur_local_blockers = begin_local_blockers;
                    }
                }
                local_blocker_counters[thread_id][row_blocker_id] = cur_local_blockers - begin_local_blockers;
            }
        }
        for (uint16_t row_blocker_id = 0; row_blocker_id < num_blockers; row_blocker_id++)
        {
            // cur_global_blockers = global_blockers[row_blocker_id] + __sync_fetch_and_add(&global_blocker_counters[row_blocker_id], local_blocker_counters[thread_id][row_blocker_id]);
            // for (IT offset=0; offset<local_blocker_counters[thread_id][row_blocker_id]; ++offset)
            // {
            //     std::get<0>(cur_global_blockers[offset]) = std::get<0>(local_blockers[thread_id][row_blocker_id * block_width + offset]);
            //     std::get<1>(cur_global_blockers[offset]) = std::get<1>(local_blockers[thread_id][row_blocker_id * block_width + offset]);
            //     std::get<2>(cur_global_blockers[offset]) = std::get<2>(local_blockers[thread_id][row_blocker_id * block_width + offset]);
            //     // cur_global_blockers[offset] = local_blockers[thread_id][row_blocker_id * block_width + offset];
            // }
            std::memcpy(
                global_blockers[row_blocker_id] + __sync_fetch_and_add(&global_blocker_counters[row_blocker_id], local_blocker_counters[thread_id][row_blocker_id]),
                local_blockers[thread_id] + row_blocker_id * block_width,
                local_blocker_counters[thread_id][row_blocker_id] * SIZE
            );
        //     local_blocker_counters[thread_id][row_blocker_id] = 0;
        }
    }
// //     vector<pair<int, int> > scheduling_order;
//     for (uint16_t blocker_id=0; blocker_id<num_blockers; ++blocker_id)
//         scheduling_order.push_back(make_pair(-global_blocker_counters[blocker_id], blocker_id));
// sort(scheduling_order.begin(), scheduling_order.end());

    // for (uint16_t row_blocker_id=0; row_blocker_id < num_blockers; ++ row_blocker_id)
    //     cout << scheduling_order[row_blocker_id].second << " " << global_blocker_counters[scheduling_order[row_blocker_id].second] << endl;
#pragma omp parallel
    {
        uint16_t thread_id = omp_get_thread_num();
#pragma omp for reduction(+ : nnz_per_row_blocker[:num_blockers])
        for (uint16_t row_blocker_id=0; row_blocker_id < num_blockers; ++ row_blocker_id)
        {
            // uint16_t my_blocker_id = scheduling_order[row_blocker_id].second;
            doRadixSort(global_blockers[row_blocker_id],
                        global_blockers[row_blocker_id] + global_blocker_counters[row_blocker_id],
                        sorting_buffer[thread_id]);
            IT after = doMerge(global_blockers[row_blocker_id], global_blocker_counters[row_blocker_id]);
            nnz_per_row_blocker[row_blocker_id] += after;
        }
    }

    IT *cumulative_row_indices = my_malloc<IT>(num_blockers + 1);
    scan(nnz_per_row_blocker, cumulative_row_indices, (IT)(num_blockers) + 1);
    IT total_nnz = cumulative_row_indices[num_blockers];

    if (C.isEmpty())
    {
        C.make_empty();
    }
    C.rows = A.rows;
    C.cols = B.cols;

    C.colids = static_cast<IT*>(::operator new(sizeof(IT[total_nnz])));
    C.rowptr = static_cast<IT*>(::operator new(sizeof(IT[C.rows+1])));
    C.values = static_cast<NT*>(::operator new(sizeof(NT[total_nnz])));

    C.rowptr[0] = 0;

#pragma omp parallel for
    for (uint16_t row_blocker_id = 0; row_blocker_id < num_blockers; ++row_blocker_id)
    {
        IT base = cumulative_row_indices[row_blocker_id];
        // auto space_addr = global_blockers[row_blocker_id];
        TripleNode* this_blocker = global_blockers[row_blocker_id];
        for (IT i = 0; i < nnz_per_row_blocker[row_blocker_id]; ++i)
        {
            ++nnz_by_row[std::get<0>(this_blocker[i])];
            C.colids[base+i] = std::get<1>(this_blocker[i]);
            C.values[base+i] = std::get<2>(this_blocker[i]);
            // ++nnz_by_row[std::get<0>(space_addr[index])];
            // C.colids[base + index] = std::get<1>(space_addr[index]);
            // C.values[base + index] = std::get<2>(space_addr[index]);
        }
    }

    scan(nnz_by_row, C.rowptr, C.rows + 1);
    C.nnz = total_nnz;

    my_free<IT>(flops_by_row_blockers);
    my_free<IT>(nnz_by_row);
    my_free<IT>(nnz_per_row_blocker);
    my_free<IT>(cumulative_row_indices);

    for (uint16_t row_blocker_id = 0; row_blocker_id < num_blockers; ++row_blocker_id)
    {
        my_free<TripleNode>(global_blockers[row_blocker_id]);
    }
    my_free<TripleNode*>(global_blockers);
    for (uint16_t thread_id=0; thread_id<nthreads; ++thread_id)
    {
        my_free<TripleNode>(local_blockers[thread_id]);
        my_free<IT>(local_blocker_counters[thread_id]);
    }
    my_free<TripleNode*>(local_blockers);
    my_free<IT*>(local_blocker_counters);
}

template <typename IT, typename NT>
void OuterSpGEMM(const CSC<IT, NT>& A, const CSR<IT, NT>& B, CSR<IT, NT>& C, int nblockers, int nblockchars, int partition)
{
    // unsigned nodes = numa_num_configured_nodes();
    // unsigned cores = numa_num_configured_cpus();
    // unsigned cores_per_node = cores / nodes;
    // omp_set_num_threads(cores_per_node);
//     cout << "Printing" << endl;
// #pragma omp parallel
//     {
// #pragma omp critical
//         cout << "executing: cpu_id-> " << sched_getcpu() << "(" << omp_get_num_threads() << ") numa id-> " << numa_node_of_cpu(sched_getcpu()) << " partition -> " << partition << endl;
//     }
    OuterSpGEMM_stage(A, B, 0, A.cols, C, nblockers, nblockchars, partition);
}

