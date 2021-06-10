#ifndef MASKED_SPGEMM_MASKED_HEAP_ALGORITHM_INSPECT_H
#define MASKED_SPGEMM_MASKED_HEAP_ALGORITHM_INSPECT_H

template<size_t Inspect = 0>
struct MaskedHeap {
    template<class IT, class NT, bool Complemented = false>
    class Impl {
        static_assert(Complemented == false);

    private:
        Heap<IT> _heap;

    public:
        inline const static bool CALC_MAX_ROW_SIZE_A = true;
        inline const static bool CALC_MAX_ROW_SIZE_M = false;

        Impl(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) : _heap(maxRowSizeA) {};

        [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() { return _heap.getMemoryRequirement(); }

        [[nodiscard]] Heap<IT> &getSymbolicAccumulator() { return _heap; }

        [[nodiscard]] Heap<IT> &getNumericAccumulator() { return _heap; }

        [[gnu::always_inline]]
        void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
            IT maskIdx = M.rowptr[row];
            const IT maskEnd = M.rowptr[row + 1];
            assert(maskIdx != maskEnd);

            // Make initial heap for the row
            initHeap(A, B, M, row, maskIdx, maskEnd);

            IT currRowNvals = 0;
            IT prevKey = std::numeric_limits<IT>::max();
            // Traverse the heaps
            while (!_heap.isEmpty()) {
                auto &hentry = _heap.top();

                while (maskIdx < maskEnd && hentry.key > M.colids[maskIdx]) { ++maskIdx; }
                if (maskIdx >= maskEnd) {
                    _heap.clear();
                    break;
                }

                if (hentry.key == M.colids[maskIdx] && prevKey != hentry.key) {
                    prevKey = hentry.key;
                    currRowNvals++;
                }

                insertNext(A, B, M, maskIdx, maskEnd, hentry);
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

            initHeap(A, B, M, row, maskIdx, maskEnd);

            IT prevKey = std::numeric_limits<IT>::max();
            --currValue, --currColId;
            // Traverse the heaps
            while (!_heap.isEmpty()) {
                auto &hentry = _heap.top();

                while (maskIdx < maskEnd && hentry.key > M.colids[maskIdx]) { ++maskIdx; }
                if (maskIdx >= maskEnd) {
                    _heap.clear();
                    break;
                }

                if (hentry.key == M.colids[maskIdx]) {
                    NT value = multop(A.values[hentry.runr], B.values[hentry.loc]);

                    // Use short circuiting
                    if (prevKey == hentry.key) {
                        *currValue = addop(value, *currValue);
                    } else {
                        prevKey = hentry.key;
                        *(++currValue) = value;
                        *(++currColId) = hentry.key;
                    }
                }

                insertNext(A, B, M, maskIdx, maskEnd, hentry);
            }
            ++currValue, ++currColId;
        }

        void initHeap(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row,
                      IT maskIdx, const IT maskEnd) {
            for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; ++j) {
                IT inner = A.colids[j];
                IT npins = B.rowptr[inner + 1] - B.rowptr[inner];
                if (npins == 0) { continue; }

                IT loc = B.rowptr[inner];
                IT key = B.colids[B.rowptr[inner]];

                switch (Inspect) {
                    case 0: {
                        _heap.append(key, j, loc);
                        break;
                    }

                    case 1: {
                        // Before pushing the entry back to the queue, remove elements that are < than current mask element
                        while (key < M.colids[maskIdx] && (loc + 1 < B.rowptr[inner + 1])) { key = B.colids[++loc]; }

                        if (loc < B.rowptr[inner + 1]) { _heap.append(key, j, loc); }
                        break;
                    }

                    case std::numeric_limits<std::size_t>::max(): {
                        // Find the first match in the intersection of the mask column and the A column
                        IT maskIdxCopy = maskIdx;

                        while (true) {
                            if (key < M.colids[maskIdx]) {
                                if (++loc >= B.rowptr[inner + 1]) { break; }
                                key = B.colids[loc];
                            } else if (key > M.colids[maskIdx]) {
                                if (++maskIdx == maskEnd) { break; }
                            } else {
                                _heap.append(key, j, loc);
                                break;
                            }
                        }

                        maskIdx = maskIdxCopy;
                        break;
                    } // end size_t::max()

                    default: {
                        auto maskStop = std::min<std::size_t>(maskIdx + Inspect, maskEnd);

                        // Find the first match in the intersection of the mask column and the A column
                        IT maskIdxCopy = maskIdx;

                        while (true) {
                            if (key < M.colids[maskIdx]) {
                                if (++loc >= B.rowptr[inner + 1]) { break; }
                                key = B.colids[loc];
                            } else if (key > M.colids[maskIdx]) {
                                if (++maskIdx == maskStop) {
                                    if (maskIdx != maskEnd) { _heap.append(key, j, loc); }
                                    break;
                                }
                            } else {
                                _heap.append(key, j, loc);
                                break;
                            }
                        }

                        maskIdx = maskIdxCopy;
                    } // end default
                }
            }

            _heap.make();
        }

        void insertNext(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                        IT maskIdx, const IT maskEnd, typename Heap<IT>::EntryT &hentry) {
            IT inner = A.colids[hentry.runr];

            switch (Inspect) {
                case 0: {
                    if (++hentry.loc < B.rowptr[inner + 1]) {
                        hentry.key = B.colids[hentry.loc];
                        _heap.sinkRoot();
                    } else {
                        _heap.pop();
                    }
                    break;
                }

                case 1: {
                    // Before pushing the entry back to the queue, remove elements that are < than current mask element
                    while (++hentry.loc < B.rowptr[inner + 1]) {
                        hentry.key = B.colids[hentry.loc];
                        if (hentry.key >= M.colids[maskIdx]) { break; }
                    }

                    if (hentry.loc < B.rowptr[inner + 1]) { _heap.sinkRoot(); } else { _heap.pop(); }
                    break;
                }

                case std::numeric_limits<size_t>::max(): {
                    // Check if we are done with the current row from B, and if we are not move to the next element.
                    if (++hentry.loc >= B.rowptr[inner + 1]) {
                        _heap.pop();
                        return;
                    }
                    hentry.key = B.colids[hentry.loc];

                    // Find the first match in the intersection of
                    // the mask column (starting with maskIdx) and the A column (starting with hentry.loc)
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
                    break;
                } // end size_t::max()

                default: {
                    IT maskStop = std::min<std::size_t>(maskIdx + Inspect, maskEnd);

                    // Check if we are done with the current row from B, and if we are not move to the next element.
                    if (++hentry.loc >= B.rowptr[inner + 1]) {
                        _heap.pop();
                        return;
                    }
                    hentry.key = B.colids[hentry.loc];
                    if (Inspect == 0) {
                        _heap.sinkRoot();
                        return;
                    }

                    // Find the first match in the intersection of
                    // the mask column (starting with maskIdx) and the A column (starting with hentry.loc)
                    while (true) {
                        if (hentry.key < M.colids[maskIdx]) {
                            if (++hentry.loc < B.rowptr[inner + 1]) {
                                hentry.key = B.colids[hentry.loc];
                            } else {
                                _heap.pop();
                                break;
                            }
                        } else if (hentry.key > M.colids[maskIdx]) {
                            if (++maskIdx == maskStop) {
                                if (maskIdx == maskEnd) { _heap.pop(); }
                                else { _heap.sinkRoot(); }
                                break;
                            }
                        } else {
                            // put the merge heap in the valid state again
                            _heap.sinkRoot();
                            break;
                        }
                    }
                    break;
                } // end default
            }
        }
    };
};

template<class IT, class NT, bool Complemented = false>
using MaskedHeapNaive = MaskedHeap<0>::Impl<IT, NT, Complemented>;

template<class IT, class NT, bool Complemented = false>
using MaskedHeapBasic = MaskedHeap<1>::Impl<IT, NT, Complemented>;

template<class IT, class NT, bool Complemented = false>
using MaskedHeapDot = MaskedHeap<std::numeric_limits<std::size_t>::max()>::Impl<IT, NT, Complemented>;

#endif //MASKED_SPGEMM_MASKED_HEAP_ALGORITHM_INSPECT_H
