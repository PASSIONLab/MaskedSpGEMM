#ifndef MASKED_SPGEMM_HEAP_MULT_GENERIC_H
#define MASKED_SPGEMM_HEAP_MULT_GENERIC_H

#include <algorithm>

#include "CSR.h"

// TODO: move to a separate file
namespace tmp {

/**
 ** Count flop of SpGEMM between A and B in CSR format
 **/
template<typename IT, typename NT>
long long int getFlop(const CSR<IT, NT> &A, const CSR<IT, NT> &B, IT *maxnnzc) {
    long long int flop = 0; // total flop (multiplication) needed to generate C

#pragma omp parallel for reduction(+:flop)
    for (IT i = 0; i < A.rows; ++i) {
        long long int locmax = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
            IT inner = A.colids[j];
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];
            locmax += npins;
        }
        maxnnzc[i] = locmax;
        flop += locmax;
    }
    return flop * 2;
}

}

namespace heap {

template<class RandomAccessIterator, class SizeT>
[[gnu::always_inline]]
void make(RandomAccessIterator heap, SizeT size) {
    std::make_heap(heap, heap + size);
}

template<class RandomAccessIterator, class SizeT>
[[gnu::always_inline]]
void pop(RandomAccessIterator heap, SizeT &size) {
    std::pop_heap(heap, heap + size);
    size--;
}

template<class RandomAccessIterator, class SizeT>
[[gnu::always_inline]]
void sinkRoot(RandomAccessIterator heap, SizeT size) {
    std::pop_heap(heap, heap + size);
    std::push_heap(heap, heap + size);
}

}

namespace rowAlg {

struct HeapBase {

    const static bool masked = false;

    template<class IT, class NT>
    static IT
    estimateResultSize(IT rowBeginIdx, IT rowEndIdx, IT *maxnnzc,
                       const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
        return std::accumulate(maxnnzc + rowBeginIdx, maxnnzc + rowEndIdx, 0);
    }

    template<class IT, class NT>
    static HeapEntry<IT, void> *allocateAuxiliaryMemory(IT rowBeginIdx, IT rowEndIdx, IT *maxnnzc,
                                                        const CSR<IT, NT> &A, const CSR<IT, NT> &B,
                                                        const CSR<IT, NT> &M) {
        IT threadHeapSize = 0;
        for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
            IT rownnz = A.rowptr[i + 1] - A.rowptr[i];
            if (rownnz > threadHeapSize) { threadHeapSize = rownnz; }
        }
        return my_malloc<HeapEntry<IT, void>>(threadHeapSize);
    };
};

struct BasicHeap : HeapBase {

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, HeapEntry<IT, void> *mergeheap,
                    IT &threadNvals) {
        // Make initial heap for the row
        IT currRowNvals = 0;
        IT hsize = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            mergeheap[hsize].loc = B.rowptr[inner];
            mergeheap[hsize].runr = j;                // the pointer to A.colid's is the run-rank
            mergeheap[hsize++].key = B.colids[B.rowptr[inner]];    // B's first colid is the first key
        }

        heap::make(mergeheap, hsize);

        // Traverse the heaps
        while (hsize > 0) {
            auto &hentry = mergeheap[0];
            NT value = multop(A.values[hentry.runr], B.values[hentry.loc]);

            // Use short circuiting
            if ((currRowNvals > 0) && *prevColIdC == hentry.key) {
                *prevValueC = addop(value, *prevValueC);
            } else {
                *(++prevValueC) = value;
                *(++prevColIdC) = hentry.key;
                currRowNvals++;
            }

            IT inner = A.colids[hentry.runr];

            // If still unused nonzeros exists in A(:,colind), insert the next nonzero to the heap
            if (++hentry.loc < B.rowptr[inner + 1]) {
                hentry.key = B.colids[hentry.loc];
                heap::sinkRoot(mergeheap, hsize);
            } else {
                heap::pop(mergeheap, hsize);
            }
        }

        rowNvals[i] = currRowNvals;
        threadNvals += currRowNvals;
    }

};

template<size_t threshold>
struct HeapLinear : HeapBase {

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, HeapEntry<IT, void> *mergeheap,
                    IT &threadNvals) {
        // Make initial heap for the row
        IT currRowNvals = 0;
        IT hsize = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            mergeheap[hsize].loc = B.rowptr[inner];
            mergeheap[hsize].runr = j;                // the pointer to A.colid's is the run-rank
            mergeheap[hsize++].key = B.colids[B.rowptr[inner]];    // B's first colid is the first key
        }

        if (hsize > threshold) { heap::make(mergeheap, hsize); }

        // Traverse the heaps
        while (hsize > 0) {
            IT idx = hsize > threshold ? 0 : std::max_element(mergeheap, mergeheap + hsize) - mergeheap;
            auto &hentry = mergeheap[idx];
            NT value = multop(A.values[hentry.runr], B.values[hentry.loc]);

            // Use short circuiting
            if ((currRowNvals > 0) && *prevColIdC == hentry.key) {
                *prevValueC = addop(value, *prevValueC);
            } else {
                *(++prevValueC) = value;
                *(++prevColIdC) = hentry.key;
                currRowNvals++;
            }

            IT inner = A.colids[hentry.runr];

            // If still unused nonzeros exists in A(:,colind), insert the next nonzero to the heap
            if (++hentry.loc < B.rowptr[inner + 1]) {
                hentry.key = B.colids[hentry.loc];
                if (hsize > threshold) {
                    heap::sinkRoot(mergeheap, hsize);
                }
            } else {
                if (hsize > threshold) {
                    heap::pop(mergeheap, hsize);
                } else {
                    *(mergeheap + idx) = *(mergeheap + --hsize);
                }
            }
        }

        rowNvals[i] = currRowNvals;
        threadNvals += currRowNvals;
    }
};

struct MaskedHeapBase {
    const static bool masked = true;

    template<class IT, class NT>
    static IT estimateResultSize(IT rowBeginIdx, IT rowEndIdx, IT *maxnnzc,
                                 const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
        IT size = 0;
        for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
            size += std::min(maxnnzc[row], M.rowptr[row + 1] - M.rowptr[row]);
        }
        return size;
    }

    template<class IT, class NT>
    static HeapEntry<IT, void> *allocateAuxiliaryMemory(IT rowBeginIdx, IT rowEndIdx, IT *maxnnzc,
                                                        const CSR<IT, NT> &A, const CSR<IT, NT> &B,
                                                        const CSR<IT, NT> &M) {
        IT threadHeapSize = 0;
        for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
            IT rownnz = A.rowptr[i + 1] - A.rowptr[i];
            if (rownnz > threadHeapSize) { threadHeapSize = rownnz; }
        }
        return my_malloc<HeapEntry<IT, void>>(threadHeapSize);
    };

};

struct MaskedBasicHeap_v1 : MaskedHeapBase {

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, HeapEntry<IT, void> *mergeheap,
                    IT &threadNvals) {
        IT maskIdx = M.rowptr[i];
        IT maskEnd = M.rowptr[i + 1];

        // Make initial heap for the row
        IT currRowNvals = 0;
        IT hsize = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            mergeheap[hsize].loc = B.rowptr[inner];
            mergeheap[hsize].runr = j;                // the pointer to A.colid's is the run-rank
            mergeheap[hsize++].key = B.colids[B.rowptr[inner]];    // B's first colid is the first key
        }

        heap::make(mergeheap, hsize);

        // Traverse the heaps
        while (hsize > 0) {
            auto &hentry = mergeheap[0];

            while (maskIdx < maskEnd && hentry.key > M.colids[maskIdx]) { ++maskIdx; }
            if (maskIdx >= maskEnd) { break; }

            if (hentry.key == M.colids[maskIdx]) {
                NT value = multop(A.values[hentry.runr], B.values[hentry.loc]);

                // Use short circuiting
                if ((currRowNvals > 0) && *prevColIdC == hentry.key) {
                    *prevValueC = addop(value, *prevValueC);
                } else {
                    *(++prevValueC) = value;
                    *(++prevColIdC) = hentry.key;
                    currRowNvals++;
                }
            }

            IT inner = A.colids[hentry.runr];

            // If still unused nonzeros exists in A(:,colind), insert the next nonzero to the heap
            if (++hentry.loc < B.rowptr[inner + 1]) {
                hentry.key = B.colids[hentry.loc];
                heap::sinkRoot(mergeheap, hsize);
            } else {
                heap::pop(mergeheap, hsize);
            }
        }

        rowNvals[i] = currRowNvals;
        threadNvals += currRowNvals;
    }
};

struct MaskedBasicHeap_v2 : MaskedHeapBase {

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, HeapEntry<IT, void> *mergeheap,
                    IT &threadNvals) {
        IT maskIdx = M.rowptr[i];
        IT maskEnd = M.rowptr[i + 1];

        if (maskIdx == maskEnd) { return; }

        // Make initial heap for the row
        IT currRowNvals = 0;
        IT hsize = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            mergeheap[hsize].loc = B.rowptr[inner];
            mergeheap[hsize].runr = j;                // the pointer to A.colid's is the run-rank
            mergeheap[hsize].key = B.colids[B.rowptr[inner]];    // B's first colid is the first key

            while (mergeheap[hsize].key < M.colids[maskIdx] && (mergeheap[hsize].loc + 1 < B.rowptr[inner + 1])) {
                mergeheap[hsize].loc++;
                mergeheap[hsize].key = B.colids[mergeheap[hsize].loc];
            }

            // If we did not reach the end of B's row, add it to the heap
            if (mergeheap[hsize].loc < B.rowptr[inner + 1]) { hsize++; }
        }

        heap::make(mergeheap, hsize);

        // Traverse the heaps
        while (hsize > 0) {
            auto &hentry = mergeheap[0];

            while (maskIdx < maskEnd && hentry.key > M.colids[maskIdx]) { ++maskIdx; }
            if (maskIdx >= maskEnd) { break; }

            if (hentry.key == M.colids[maskIdx]) {
                NT value = multop(A.values[hentry.runr], B.values[hentry.loc]);

                // Use short circuiting
                if ((currRowNvals > 0) && *prevColIdC == hentry.key) {
                    *prevValueC = addop(value, *prevValueC);
                } else {
                    *(++prevValueC) = value;
                    *(++prevColIdC) = hentry.key;
                    currRowNvals++;
                }
            }

            IT inner = A.colids[hentry.runr];

            // Before pushing the entry back to the queue, remove elements that are < than current mask element
            while (++hentry.loc < B.rowptr[inner + 1]) {
                hentry.key = B.colids[hentry.loc];
                if (hentry.key >= M.colids[maskIdx]) { break; }
            }

            if (hentry.loc < B.rowptr[inner + 1]) {
                heap::sinkRoot(mergeheap, hsize);
            } else {
                heap::pop(mergeheap, hsize);
            }
        }

        rowNvals[i] = currRowNvals;
        threadNvals += currRowNvals;
    }
};

struct MaskedBasicHeap_v3 : MaskedHeapBase {

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, HeapEntry<IT, void> *mergeheap,
                    IT &threadNvals) {
        IT maskIdx = M.rowptr[i];
        IT maskEnd = M.rowptr[i + 1];

        if (maskIdx == maskEnd) { return; }

        // Make initial heap for the row
        IT currRowNvals = 0;
        IT hsize = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            mergeheap[hsize].loc = B.rowptr[inner];
            mergeheap[hsize].runr = j;                // the pointer to A.colid's is the run-rank
            mergeheap[hsize].key = B.colids[B.rowptr[inner]];    // B's first colid is the first key

            // Find the first match in the intersection of the mask column and the A column
            IT maskIdxCopy = maskIdx;

            while (true) {
                if (mergeheap[hsize].key < M.colids[maskIdx]) {
                    if (++mergeheap[hsize].loc < B.rowptr[inner + 1]) {
                        mergeheap[hsize].key = B.colids[mergeheap[hsize].loc];
                    } else {
                        break;
                    }
                } else if (mergeheap[hsize].key > M.colids[maskIdx]) {
                    if (++maskIdx == maskEnd) {
                        break;
                    }
                } else {
                    hsize++;
                    break;
                }
            }

            maskIdx = maskIdxCopy;
        }

        heap::make(mergeheap, hsize);

        // Traverse the heaps
        while (hsize > 0) {
            auto &hentry = mergeheap[0];

            while (maskIdx < maskEnd && hentry.key > M.colids[maskIdx]) { ++maskIdx; }
            if (maskIdx >= maskEnd) { break; }

            if (hentry.key == M.colids[maskIdx]) {
                NT value = multop(A.values[hentry.runr], B.values[hentry.loc]);

                // Use short circuiting
                if ((currRowNvals > 0) && *prevColIdC == hentry.key) {
                    *prevValueC = addop(value, *prevValueC);
                } else {
                    *(++prevValueC) = value;
                    *(++prevColIdC) = hentry.key;
                    currRowNvals++;
                }
            }

            IT inner = A.colids[hentry.runr];

            // Check if we are done with the current row from B, and if we are not move to the next element.
            if (++hentry.loc >= B.rowptr[inner + 1]) {
                heap::pop(mergeheap, hsize);
                continue;
            }
            hentry.key = B.colids[hentry.loc];

            // Find the first match in the intersection of
            // the mask column (starting with maskIdx) and the A column (starting with hentry.loc)
            IT maskIdxCopy = maskIdx;

            while (true) {
                if (hentry.key < M.colids[maskIdx]) {
                    if (++hentry.loc < B.rowptr[inner + 1]) {
                        hentry.key = B.colids[hentry.loc];
                    } else {
                        heap::pop(mergeheap, hsize);
                        break;
                    }
                } else if (hentry.key > M.colids[maskIdx]) {
                    if (++maskIdx == maskEnd) {
                        heap::pop(mergeheap, hsize);
                        break;
                    }
                } else {
                    // put the merge heap in the valid state again
                    heap::sinkRoot(mergeheap, hsize);
                    break;
                }
            }

            maskIdx = maskIdxCopy;
        }

        rowNvals[i] = currRowNvals;
        threadNvals += currRowNvals;
    }
};

struct MaskIndexedBase {
    const static bool masked = true;

    template<class IT, class NT>
    static IT estimateResultSize(IT rowBeginIdx, IT rowEndIdx, IT *maxnnzc,
                                 const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
        IT size = 0;
        for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
            size += std::min(maxnnzc[row], M.rowptr[row + 1] - M.rowptr[row]);
        }
        return size;
    }

    template<class IT, class NT>
    static bool *allocateAuxiliaryMemory(IT rowBeginIdx, IT rowEndIdx, IT *maxnnzc,
                                         const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
        IT flagsSize = 0;
        for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
            IT maxMRow = M.rowptr[i + 1] - M.rowptr[i];
            if (maxMRow > flagsSize) { flagsSize = maxMRow; }
        }
        return my_malloc<bool>(flagsSize);
    };

};


struct MaskIndexed_v1 : MaskIndexedBase {

    const static bool masked = true;

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, bool *flags,
                    IT &threadNvals) {
        IT maskBegin = M.rowptr[i];
        const IT maskEnd = M.rowptr[i + 1];
        const IT maskSize = maskEnd - maskBegin;

        prevColIdC++;
        prevValueC++;

        std::fill(flags, flags + maskSize, false);

        // Iterate though nonzeros in the A's current row
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; j++) {
            const IT inner = A.colids[j];
            IT loc = B.rowptr[inner];
            IT key = A.colids[loc];

            if (loc == B.rowptr[inner + 1]) { continue; }

            IT maskIdx = maskBegin;

            // Find the intersection between the mask's row and the A's row
            while (true) {
                if (key < M.colids[maskIdx]) {
                    if (++loc < B.rowptr[inner + 1]) { key = B.colids[loc]; } else { break; }
                } else if (key > M.colids[maskIdx]) {
                    if (++maskIdx == maskEnd) { break; }
                } else {
                    // colid is found in both arrays
                    const auto idx = maskIdx - maskBegin;
                    const NT value = multop(A.values[j], B.values[loc]);

                    if (!flags[idx]) {
                        prevValueC[idx] = value;
                        flags[idx] = true;
                    } else {
                        prevValueC[idx] = addop(prevValueC[idx], value);
                    }

                    if (++loc < B.rowptr[inner + 1]) { key = B.colids[loc]; } else { break; }
                    if (++maskIdx == maskEnd) { break; }
                }
            }
        }

        /* Remove empty values the destination arrays and set row IDs */
        size_t dst = 0;
        for (size_t src = 0; src < maskSize; src++) {
            if (flags[src]) {
                prevColIdC[dst] = M.colids[maskBegin + src];
                prevValueC[dst] = prevValueC[src];
                dst++;
            }
        }

        prevColIdC += dst - 1;
        prevValueC += dst - 1;

        rowNvals[i] = dst;
        threadNvals += dst;
    }
};

struct MaskIndexed_v2 : MaskIndexedBase {

    const static bool masked = true;

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, bool *flags,
                    IT &threadNvals) {
        IT maskBegin = M.rowptr[i];
        const IT maskEnd = M.rowptr[i + 1];
        const IT maskSize = maskEnd - maskBegin;

        // Since prev***C point to the previous element, increment them
        prevColIdC++;
        prevValueC++;

        std::fill(flags, flags + maskSize, false);

        // Iterate though nonzeros in the A's current row
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; j++) {
            const IT inner = A.colids[j];
            IT loc = B.rowptr[inner];
            IT key = A.colids[loc];

            if (loc == B.rowptr[inner + 1]) { continue; }

            // Find the intersection between the mask's row and the A's row
            for (IT maskIdx = maskBegin; maskIdx < maskEnd; maskIdx++) {
                while (key < M.colids[maskIdx]) {
                    if (++loc < B.rowptr[inner + 1]) { key = B.colids[loc]; } else { goto outerLoopBreak; }
                }

                if (key == M.colids[maskIdx]) {
                    // colid is found in both arrays
                    const auto idx = maskIdx - maskBegin;
                    const NT value = multop(A.values[j], B.values[loc]);

                    if (!flags[idx]) {
                        prevValueC[idx] = value;
                        flags[idx] = true;
                    } else {
                        prevValueC[idx] = addop(prevValueC[idx], value);
                    }
                    if (++loc < B.rowptr[inner + 1]) { key = B.colids[loc]; } else { break; }
                }

            }
            outerLoopBreak:
            continue;
        }

        /* Remove empty values the destination arrays and set row IDs */
        size_t dst = 0;
        for (size_t src = 0; src < maskSize; src++) {
            if (flags[src]) {
                prevColIdC[dst] = M.colids[maskBegin + src];
                prevValueC[dst] = prevValueC[src];
                dst++;
            }
        }

        prevColIdC += dst - 1;
        prevValueC += dst - 1;

        rowNvals[i] = dst;
        threadNvals += dst;
    }
};

struct MaskIndexed_v3 : MaskIndexedBase {

    const static bool masked = true;

    template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    static void row(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT i,
                    IT *rowNvals, IT *&prevColIdC, NT *&prevValueC, bool *flags,
                    IT &threadNvals) {
        const auto maskBegin = &M.colids[M.rowptr[i]];
        const auto maskEnd = &M.colids[M.rowptr[i + 1]];
        const auto maskSize = maskEnd - maskBegin;

        prevColIdC++;
        prevValueC++;

        std::fill(flags, flags + maskSize, false);

        // Iterate though nonzeros in the A's current row
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; j++) {
            const IT inner = A.colids[j];
            auto colIdsIt = &B.colids[B.rowptr[inner]];
            const auto colIdsBegin = &B.colids[B.rowptr[inner]];
            const auto colIdsEnd = &B.colids[B.rowptr[inner + 1]];

            auto maskIt = maskBegin;

            if (colIdsIt == colIdsEnd) { continue; }

            // Find the intersection between the mask's row and the A's row
            while (true) {
                if (*colIdsIt < *maskIt) {
                    if (++colIdsIt == colIdsEnd) { break; }
                } else if (*colIdsIt > *maskIt) {
                    if (++maskIt == maskEnd) { break; }
                } else {
                    // colid is found in both arrays
                    const auto idx = maskIt - maskBegin;
                    const NT value = multop(A.values[j], B.values[B.rowptr[inner] + colIdsIt - colIdsBegin]);

                    if (!flags[idx]) {
                        prevValueC[idx] = value;
                        flags[idx] = true;
                    } else {
                        prevValueC[idx] = addop(prevValueC[idx], value);
                    }

                    if (++colIdsIt >= colIdsEnd) { break; }
                    if (++maskIt == maskEnd) { break; }
                }
            }
        }

        /* Remove empty values the destination arrays and set row IDs */
        size_t dst = 0;
        for (size_t src = 0; src < maskSize; src++) {
            if (flags[src]) {
                prevColIdC[dst] = maskBegin[src];
                prevValueC[dst] = prevValueC[src];
                dst++;
            }
        }

        prevColIdC += dst - 1;
        prevValueC += dst - 1;

        rowNvals[i] = dst;
        threadNvals += dst;
    }
};

}

template<bool masked, class RowAlgorithm, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HeapSpGEMMImpl(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads) {
    static_assert(masked == RowAlgorithm::masked || masked, "Row algorithm does not support mask.");
    static_assert(masked == RowAlgorithm::masked || !masked, "Row algorithm is used for masked computation.");

    if (numThreads == 0) {
#pragma omp parallel
#pragma omp single
        numThreads = omp_get_num_threads();
    }

    if (!C.isEmpty()) { C.make_empty(); }
    C.rows = A.rows;
    C.cols = B.cols;

    // Load-balancing Thread Scheduling
    IT *maxnnzc = my_malloc<IT>(A.rows);
    long long int flops = tmp::getFlop(A, B, maxnnzc) / 2;

    IT flopsPerThread = flops / numThreads; // amount of work that will be assigned to each thread

    IT *rowStart = my_malloc<IT>(A.rows); //start index in the global array for storing ith column of C
    IT *rowNvals = my_malloc<IT>(A.rows); // number of nonzeros in each each column in C

    rowStart[0] = 0;

    // Global space used to store result
    IT *threadsNvals = my_malloc<IT>(numThreads);

    // Parallelized version
    scan(maxnnzc, rowStart, A.rows);

    // ************************ Numeric Phase *************************************
#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();

        // @formatter:off
        IT rowBegin = thisThread != 0 ? (lower_bound(rowStart, rowStart + A.rows, flopsPerThread * thisThread)) - rowStart : 0;
        IT rowEnd = thisThread != numThreads - 1 ? (lower_bound(rowStart, rowStart + A.rows, flopsPerThread * (thisThread + 1))) - rowStart : A.rows;
        // @formatter:on

        IT localsum = RowAlgorithm::estimateResultSize(rowBegin, rowEnd, maxnnzc, A, B, M);

        // We need +1 even though the first element of the array is never accessed.
        // However, the first element may be prefetched so we have to allocate it together with the rest of the array.
        IT *colIdsLocalMem = my_malloc<IT>(localsum + 1);
        NT *valuesLocalMem = my_malloc<NT>(localsum + 1);
        IT *prevColIdC = colIdsLocalMem;
        NT *prevValueC = valuesLocalMem;

        auto auxMemory = RowAlgorithm::allocateAuxiliaryMemory(rowBegin, rowEnd, maxnnzc, A, B, M);
        IT threadNvals = 0;

        // Iterate through all rows in A
        for (IT i = rowBegin; i < rowEnd; ++i) {
            RowAlgorithm::row(A, B, M, multop, addop, i, rowNvals, prevColIdC, prevValueC, auxMemory, threadNvals);
        }
        threadsNvals[thisThread] = threadNvals;
        my_free(auxMemory);

#pragma omp barrier
#pragma omp master
        {
            C.rowptr = my_malloc<IT>(C.rows + 1);
            C.rowptr[0] = 0;

            C.nnz = std::accumulate(threadsNvals, threadsNvals + numThreads, IT(0));;
            C.colids = my_malloc<IT>(C.nnz);
            C.values = my_malloc<NT>(C.nnz);
        }

        IT rowPtrOffset = std::accumulate(threadsNvals, threadsNvals + thisThread, IT(0));
#pragma omp barrier

        // set rowptr in C for local rows
        for (IT i = rowBegin; i < rowEnd; ++i) {
            C.rowptr[i] = rowPtrOffset;
            rowPtrOffset += rowNvals[i];
        }
        if (thisThread == numThreads - 1) { C.rowptr[C.rows] = rowPtrOffset; }

        // copy local values to C
        copy(colIdsLocalMem + 1, colIdsLocalMem + threadNvals + 1, C.colids + C.rowptr[rowBegin]);
        copy(valuesLocalMem + 1, valuesLocalMem + threadNvals + 1, C.values + C.rowptr[rowBegin]);

        my_free<IT>(colIdsLocalMem);
        my_free<NT>(valuesLocalMem);
    }

    my_free<IT>(maxnnzc);
    my_free<IT>(rowStart);
    my_free<IT>(rowNvals);
}

template<class RowAlgorithm, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HeapSpGEMM(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C,
                MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    HeapSpGEMMImpl<false, RowAlgorithm>(A, B, C, CSR<IT, NT>{}, multop, addop, numThreads);
}

template<class RowAlgorithm, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HeapSpGEMM(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    HeapSpGEMMImpl<true, RowAlgorithm>(A, B, C, M, multop, addop, numThreads);
}


#endif //MASKED_SPGEMM_HEAP_MULT_GENERIC_H
