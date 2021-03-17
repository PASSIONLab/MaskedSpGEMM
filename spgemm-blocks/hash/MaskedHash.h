#ifndef MASKED_SPGEMM_MASKED_HASH_H
#define MASKED_SPGEMM_MASKED_HASH_H

#include "../common.h"

template<class IT, class NT>
class MaskedHash {
private:
    HashAccumulator<IT, void, void> _symbolicAccumulator;
    HashAccumulator<IT, NT, bool> _numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = true;

    MaskedHash(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM)
            : _symbolicAccumulator(maxRowSizeM), _numericAccumulator(maxRowSizeM) {};

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    [[nodiscard]] HashAccumulator<IT, void, void> &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] HashAccumulator<IT, NT, bool> &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the hashmap
        _symbolicAccumulator.resize(maskEnd - maskBegin);
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _symbolicAccumulator.insert(*maskIt);
        }

        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                bool erased = _symbolicAccumulator.erase(B.colids[k]);
                if (erased) { ++currRowNvals; }
            }
        }

        _symbolicAccumulator.reset();
        rowNvals[row] = currRowNvals;
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the hashmap
        _numericAccumulator.resize(maskEnd - maskBegin);
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _numericAccumulator.insert(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                auto idx = _numericAccumulator.find(B.colids[k]);
                if (idx == HashAccumulator<IT, NT, bool>::NOT_FOUND) { continue; }

                auto &entry = _numericAccumulator[idx];

                if (entry.value2) {
                    entry.value1 = addop(entry.value1, multop(A.values[j], B.values[k]));
                } else {
                    entry.value1 = multop(A.values[j], B.values[k]);
                    entry.value2 = true;
                }
            }
        }

        // Copy the values from the hash table to the output
        _numericAccumulator.gather(currColId, currValue);
    }
};

#endif //MASKED_SPGEMM_MASKED_HASH_H
