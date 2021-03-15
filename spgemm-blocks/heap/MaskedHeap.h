#ifndef MASKED_SPGEMM_MASKED_HEAP_H
#define MASKED_SPGEMM_MASKED_HEAP_H

#include "Heap.h"

template<class IT, class NT>
class MaskedHeap {
private:
    Heap<IT> _heap;
    Heap<IT> &_symbolicAccumulator;
    Heap<IT> &_numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = true;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;

    explicit MaskedHeap(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM)
            : _heap(maxRowSizeA), _symbolicAccumulator(_heap), _numericAccumulator(_heap) {};

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() { return _heap.getMemoryRequirement(); }

    [[nodiscard]] Heap<IT> &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] Heap<IT> &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
        IT maskIdx = M.rowptr[row];
        IT maskEnd = M.rowptr[row + 1];
        assert(maskIdx != maskEnd);

        // Make initial heap for the row
        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            IT loc = B.rowptr[inner];
            IT key = B.colids[B.rowptr[inner]];

            while (key < M.colids[maskIdx] && (loc + 1 < B.rowptr[inner + 1])) {
                ++loc;
                key = B.colids[loc];
            }

            if (loc < B.rowptr[inner + 1]) { _heap.append(key, j, loc); }
        }

        _heap.make();

        IT prevKey = std::numeric_limits<IT>::max();
        // Traverse the heaps
        while (!_heap.isEmpty()) {
            auto &hentry = _heap.top();

            while (maskIdx < maskEnd && hentry.key > M.colids[maskIdx]) { ++maskIdx; }
            if (maskIdx >= maskEnd) { break; }

            if (hentry.key == M.colids[maskIdx] && prevKey != hentry.key) {
                prevKey = hentry.key;
                currRowNvals++;
            }

            IT inner = A.colids[hentry.runr];

            // Before pushing the entry back to the queue, remove elements that are < than current mask element
            while (++hentry.loc < B.rowptr[inner + 1]) {
                hentry.key = B.colids[hentry.loc];
                if (hentry.key >= M.colids[maskIdx]) { break; }
            }

            if (hentry.loc < B.rowptr[inner + 1]) {
                _heap.sinkRoot();
            } else {
                _heap.pop();
            }
        }

        rowNvals[row] = currRowNvals;
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
        IT maskIdx = M.rowptr[row];
        IT maskEnd = M.rowptr[row + 1];
        assert(maskIdx != maskEnd);

        // Make initial heap for the row
        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            IT loc = B.rowptr[inner];
            IT key = B.colids[B.rowptr[inner]];

            while (key < M.colids[maskIdx] && (loc + 1 < B.rowptr[inner + 1])) {
                ++loc;
                key = B.colids[loc];
            }

            if (loc < B.rowptr[inner + 1]) { _heap.append(key, j, loc); }
        }

        _heap.make();

        IT prevKey = std::numeric_limits<IT>::max();
        --currValue, --currColId;
        // Traverse the heaps
        while (!_heap.isEmpty()) {
            auto &hentry = _heap.top();

            while (maskIdx < maskEnd && hentry.key > M.colids[maskIdx]) { ++maskIdx; }
            if (maskIdx >= maskEnd) { break; }

            if (hentry.key == M.colids[maskIdx]) {
                NT value = multop(A.values[hentry.runr], B.values[hentry.loc]);

                // Use short circuiting
                if (prevKey == hentry.key) {
                    *currValue = addop(value, *currValue);
                } else {
                    prevKey = hentry.key;
                    currRowNvals++;
                    *(++currValue) = value;
                    *(++currColId) = hentry.key;
                }
            }

            IT inner = A.colids[hentry.runr];

            // Before pushing the entry back to the queue, remove elements that are < than current mask element
            while (++hentry.loc < B.rowptr[inner + 1]) {
                hentry.key = B.colids[hentry.loc];
                if (hentry.key >= M.colids[maskIdx]) { break; }
            }

            if (hentry.loc < B.rowptr[inner + 1]) {
                _heap.sinkRoot();
            } else {
                _heap.pop();
            }
        }
        ++currValue, ++currColId;
    }
};

#endif //MASKED_SPGEMM_MASKED_HEAP_H
