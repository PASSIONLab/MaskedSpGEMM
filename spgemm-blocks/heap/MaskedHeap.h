#ifndef MASKED_SPGEMM_MASKED_HEAP_H
#define MASKED_SPGEMM_MASKED_HEAP_H

#include "Heap.h"

template<class IT, class NT>
class MaskedHeap_v1 {
private:
    Heap<IT> _heap;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = true;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;

    MaskedHeap_v1(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) : _heap(maxRowSizeA) {};

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() { return _heap.getMemoryRequirement(); }

    [[nodiscard]] Heap<IT> &getSymbolicAccumulator() { return _heap; }

    [[nodiscard]] Heap<IT> &getNumericAccumulator() { return _heap; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
        IT maskIdx = M.rowptr[row];
        const IT maskEnd = M.rowptr[row + 1];
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
        const IT maskEnd = M.rowptr[row + 1];
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

template<class IT, class NT>
class MaskedHeap_v2 {
private:
    Heap<IT> _heap;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = true;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;

    MaskedHeap_v2(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) : _heap(maxRowSizeA) {};

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() { return _heap.getMemoryRequirement(); }

    [[nodiscard]] Heap<IT> &getSymbolicAccumulator() { return _heap; }

    [[nodiscard]] Heap<IT> &getNumericAccumulator() { return _heap; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
        IT maskIdx = M.rowptr[row];
        const IT maskEnd = M.rowptr[row + 1];
        assert(maskIdx != maskEnd);

        // Make initial heap for the row
        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; ++j) {   // For all the nonzeros of the ith column
            IT inner = A.colids[j];                // get the col id of A (or row id of B)
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];    // get the number of nzs in B's row
            if (npins == 0) { continue; }

            IT loc = B.rowptr[inner];
            IT key = B.colids[B.rowptr[inner]];

            // Find the first match in the intersection of the mask column and the A column
            IT maskIdxCopy = maskIdx;

            while (true) {
                if (key < M.colids[maskIdx]) {
                    if (++loc < B.rowptr[inner + 1]) {
                        key = B.colids[loc];
                    } else {
                        break;
                    }
                } else if (key > M.colids[maskIdx]) {
                    if (++maskIdx == maskEnd) {
                        break;
                    }
                } else {
                    _heap.append(key, j, loc);
                    break;
                }
            }

            maskIdx = maskIdxCopy;
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

            // Check if we are done with the current row from B, and if we are not move to the next element.
            if (++hentry.loc >= B.rowptr[inner + 1]) {
                _heap.pop();
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
                        _heap.pop();
                        break;
                    }
                } else if (hentry.key > M.colids[maskIdx]) {
                    if (++maskIdx == maskEnd) {
                        _heap.pop();
                        break;
                    }
                } else {
                    // put the merge heap in the valid state again
                    _heap.sinkRoot();
                    break;
                }
            }

            maskIdx = maskIdxCopy;
        }
        ++currValue, ++currColId;
    }
};

#endif //MASKED_SPGEMM_MASKED_HEAP_H