#ifndef MASKED_SPGEMM_MASKED_HASH_H
#define MASKED_SPGEMM_MASKED_HASH_H

#include "../common.h"

template<class IT, class NT>
class MaskedHash {
private:
    HashAccumulator<IT, void, void> _symbolicAccumulator;
    HashAccumulator<IT, NT, bool> _numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_UPPER_BOUND_SIZE_C = false;
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = true;

    explicit MaskedHash(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) {};

    std::tuple<size_t, size_t> getMemoryRequirement(IT maxRowUpperBoundSizeC, IT maxRowSizeA, IT maxRowSizeM) {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement(maxRowSizeM);
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement(maxRowSizeM);

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

// region symbolic and numeric free functions

template<class IT, class NT>
auto SymbolicMaskedHashAllocateAuxiliaryMemory(IT rowBeginIdx, IT rowEndIdx, IT *flopsPerRow,
                                               const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
    IT largestRowM = 0;
    for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
        IT rownnz = M.rowptr[i + 1] - M.rowptr[i];
        if (rownnz > largestRowM) { largestRowM = rownnz; }
    }

    return HashAccumulator<IT>(largestRowM, false);
}

template<class IT, class NT>
auto NumericMaskedHashAllocateAuxiliaryMemory(IT rowBeginIdx, IT rowEndIdx, IT *flopsPerRow,
                                              const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
    IT largestRowM = 0;
    for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
        IT rownnz = M.rowptr[i + 1] - M.rowptr[i];
        if (rownnz > largestRowM) { largestRowM = rownnz; }
    }

    return HashAccumulator<IT, NT, bool>(largestRowM, true);
}

template<typename IT, typename NT>
void SymbolicMaskedHashRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT i,
                           IT *rowNvals, HashAccumulator<IT, void, void> &hashTable) {
    const auto maskBegin = &M.colids[M.rowptr[i]];
    const auto maskEnd = &M.colids[M.rowptr[i + 1]];

    // Insert all mask elements to the hashmap
    hashTable.resize(maskEnd - maskBegin);
    hashTable.reset();
    for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
        hashTable.template insert(*maskIt);
    }

    IT currRowNvals = 0;
    for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; j++) {
        IT inner = A.colids[j];
        for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
            bool erased = hashTable.erase(B.colids[k]);
            if (erased) { ++currRowNvals; }
        }
    }

    hashTable.reset();
    rowNvals[i] = currRowNvals;
}

template<typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void NumericMaskedHashRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                          MultiplyOperation multop, AddOperation addop, IT i,
                          IT *&currColId, NT *&currValue, HashAccumulator<IT, NT, bool> &hashTable) {
    const auto maskBegin = &M.colids[M.rowptr[i]];
    const auto maskEnd = &M.colids[M.rowptr[i + 1]];

    // Insert all mask elements to the hashmap
    hashTable.resize(maskEnd - maskBegin);
    for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
        hashTable.template insert(*maskIt);
    }

    for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; j++) {
        IT inner = A.colids[j];
        for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
            auto idx = hashTable.find(B.colids[k]);
            if (idx == HashAccumulator<IT, NT, bool>::NOT_FOUND) { continue; }

            auto &entry = hashTable[idx];

            if (entry.value2) {
                entry.value1 = addop(entry.value1, multop(A.values[j], B.values[k]));
            } else {
                entry.value1 = multop(A.values[j], B.values[k]);
                entry.value2 = true;
            }
        }
    }

    // Copy the values from the hash table to the output
    hashTable.gather(currColId, currValue);
}

// endregion



#endif //MASKED_SPGEMM_MASKED_HASH_H
