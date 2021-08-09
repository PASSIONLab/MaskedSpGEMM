#ifndef MASKED_SPGEMM_MASKED_HASH_H
#define MASKED_SPGEMM_MASKED_HASH_H

#include "../common.h"

namespace internal {

template<class IT, class NT, bool Sorted>
class MaskedHash {
    static_assert(Sorted == false);

public:
    inline const static bool COMPLEMENTED = false;
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = true;
    inline const static bool CALC_MAX_ROW_FLOPS = false;

private:
    using SymbolicAccumulatorT = HashAccumulator<IT, void, void>;
    using NumericAccumulatorT = HashAccumulator<IT, NT, bool>;
    SymbolicAccumulatorT _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    MaskedHash(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops) :
            _symbolicAccumulator(maxRowSizeM), _numericAccumulator(maxRowSizeM) {};

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    [[nodiscard]] SymbolicAccumulatorT &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] NumericAccumulatorT &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                     IT row, IT *rowNvals, IT &flops) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the hashmap
        _symbolicAccumulator.resize(maskEnd - maskBegin);
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _symbolicAccumulator.template insert<false>(*maskIt);
        }

        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                bool erased = _symbolicAccumulator.erase(B.colids[k]);
                if (erased) { ++currRowNvals; }
            }
        }

        _symbolicAccumulator.resetStates();
        rowNvals[row] = currRowNvals;
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue, IT &flops) {
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
                if (idx == NumericAccumulatorT::NOT_FOUND) { continue; }

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
//        _numericAccumulator.gather(currColId, currValue);
        _numericAccumulator.gather(currColId, currValue, maskBegin, maskEnd);
    }
};

template<class IT, class NT, bool Sorted>
class MaskedHashComplemented {
public:
    inline const static bool COMPLEMENTED = true;
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = true;
    inline const static bool CALC_MAX_ROW_FLOPS = true;


private:
    using SymbolicAccumulatorT = HashAccumulator<IT, void, void>;
    using NumericAccumulatorT = HashAccumulator<IT, NT, bool>;
    SymbolicAccumulatorT _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    MaskedHashComplemented(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops) :
    _symbolicAccumulator(maxRowSizeM + maxRowFlops), _numericAccumulator(maxRowSizeM + maxRowFlops) {
        _symbolicAccumulator.resize(maxRowFlops);
        _numericAccumulator.resize(maxRowFlops);
    };

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    [[nodiscard]] SymbolicAccumulatorT &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] NumericAccumulatorT &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals, IT &flops) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        _symbolicAccumulator.resize((maskEnd - maskBegin) + flops);

        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                bool inserted = _symbolicAccumulator.template insert<true>(B.colids[k]);
                if (inserted) { ++currRowNvals; }
            }
        }

        // Count the number of elements that will be discarded
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            bool found = _symbolicAccumulator.find(*maskIt) != SymbolicAccumulatorT::NOT_FOUND;
            if (found) { --currRowNvals; }
        }

        _symbolicAccumulator.resetStatesList();
        rowNvals[row] = currRowNvals;
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue, IT &flops) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Mark all elements from the mask as NOT ALLOWED (set bool field to false)
        _numericAccumulator.resize((maskEnd - maskBegin) + flops);
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _numericAccumulator.insert(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                auto idx = _numericAccumulator.findIdx(B.colids[k]);
                auto &entry = _numericAccumulator[idx];

                if (entry.key == NumericAccumulatorT::EMPTY) {
                    // If element is not found, that means it is in state ALLOWED
                    _numericAccumulator.template insert<true>(idx, B.colids[k], multop(A.values[j], B.values[k]), true);
                } else if (entry.value2 == true) {
                    // If value2 is true, it means that an element with the same key was previously initialized
                    entry.value1 = addop(entry.value1, multop(A.values[j], B.values[k]));
                }
            }
        }

        // Copy the values from the hash table to the output
        _numericAccumulator.template gatherList<Sorted>(currColId, currValue);
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _numericAccumulator.erase(*maskIt);
        }
    }
};
}

template<bool Complemented, bool Sorted>
struct MaskedHash {

    template<class IT, class NT>
    using Impl = std::conditional_t<!Complemented,
            internal::MaskedHash<IT, NT, Sorted>,
            internal::MaskedHashComplemented<IT, NT, Sorted>>;

};

#endif //MASKED_SPGEMM_MASKED_HASH_H
