#ifndef MASKED_SPGEMM_MASK_INDEXED_H
#define MASKED_SPGEMM_MASK_INDEXED_H

#include "MaskIndexedAccumulator.h"

template<class IT, class NT>
class MaskIndexed {
private:
    using SymbolicAccumulatorT = MaskIndexedAccumulator<IT, void>;
    using NumericAccumulatorT = MaskIndexedAccumulator<IT, NT>;
    SymbolicAccumulatorT _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = true;

    explicit MaskIndexed(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM)
            : _symbolicAccumulator(maxRowSizeM), _numericAccumulator(maxRowSizeM) {};

    std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    void startSymbolic(std::byte *buffer, size_t bufferSize, size_t dirty) {
        _symbolicAccumulator.setBuffer(buffer, bufferSize, dirty);
    }

    void stopSymbolic(size_t &dirty) {
        _symbolicAccumulator.releaseBuffer(dirty);
    }

    void startNumeric(std::byte *buffer, size_t bufferSize, size_t dirty) {
        _numericAccumulator.setBuffer(buffer, bufferSize, dirty);
    }

    void stopNumeric(size_t &dirty) {
        _numericAccumulator.releaseBuffer(dirty);
    }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
        const IT maskBegin = M.rowptr[row];
        const IT maskEnd = M.rowptr[row + 1];
        const IT maskSize = maskEnd - maskBegin;

        IT currRowNvals = 0;

        // Iterate though nonzeros in the A's current row
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
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

                    if (_symbolicAccumulator[idx].state == SymbolicAccumulatorT::EMPTY) {
                        _symbolicAccumulator[idx].state = SymbolicAccumulatorT::ALLOWED;
                        currRowNvals++;
                    }

                    if (++loc < B.rowptr[inner + 1]) { key = B.colids[loc]; } else { break; }
                    if (++maskIdx == maskEnd) { break; }
                }
            }
        }

        /* Remove empty values the destination arrays and set row IDs */
        _symbolicAccumulator.clearAll();

        rowNvals[row] = currRowNvals;
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
        const IT maskBegin = M.rowptr[row];
        const IT maskEnd = M.rowptr[row + 1];
        const IT maskSize = maskEnd - maskBegin;

        // Iterate though nonzeros in the A's current row
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
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

                    auto &entry = _numericAccumulator[idx];

                    if (entry.state == NumericAccumulatorT::INITIALIZED) {
                        entry.value = addop(entry.value, value);
                    } else {
                        entry.value = value;
                        entry.state = NumericAccumulatorT::INITIALIZED;
                    }

                    if (++loc < B.rowptr[inner + 1]) { key = B.colids[loc]; } else { break; }
                    if (++maskIdx == maskEnd) { break; }
                }
            }
        }

        /* Remove empty values the destination arrays and set row IDs */
        for (size_t i = 0; i < maskSize; i++) {
            auto &entry = _numericAccumulator[i];

            if (entry.state == NumericAccumulatorT::INITIALIZED) {
                *(currColId++) = M.colids[M.rowptr[row] + i];
                *(currValue++) = entry.value;
            }
        }
        _numericAccumulator.clearAll();
    }

};

#endif //MASKED_SPGEMM_MASK_INDEXED_H