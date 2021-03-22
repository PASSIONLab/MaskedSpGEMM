#ifndef MASKED_SPGEMM_MASKED_SPA_H
#define MASKED_SPGEMM_MASKED_SPA_H

#include "SparseAccumulator.h"
#include "SparseAccumulator2A.h"


template<class IT, class NT>
class MaskedSPA {
private:
    using SymbolicAccumulator = SparseAccumulator<IT, void>;
    using NumericAccumulatorT = SparseAccumulator<IT, NT>;
    SymbolicAccumulator _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;

    explicit MaskedSPA(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM)
            : _symbolicAccumulator(maxIndex), _numericAccumulator(maxIndex) {}

    [[nodiscard]]  std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    [[nodiscard]] SymbolicAccumulator &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] NumericAccumulatorT &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
//        assert(_symbolicAccumulator.isInitialized());

        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _symbolicAccumulator.setAllowed(*maskIt);
        }

        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                bool erased = _symbolicAccumulator.erase(B.colids[k]);
                if (erased) {
                    currRowNvals++;
                }
            }
        }

        // Reset - Remove all mask elements from the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _symbolicAccumulator.clear(*maskIt);
        }

        rowNvals[row] = currRowNvals;

//        assert(_symbolicAccumulator.isInitialized());
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _numericAccumulator.setAllowed(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                auto &entry = _numericAccumulator[B.colids[k]];

                if (entry.state == NumericAccumulatorT::ALLOWED) {
                    entry.value = multop(A.values[j], B.values[k]);
                    entry.state = NumericAccumulatorT::INITIALIZED;
                } else if (entry.state == NumericAccumulatorT::INITIALIZED) {
                    entry.value = addop(entry.value, multop(A.values[j], B.values[k]));
                }
            }
        }

        // Copy the values from the hash table to the output
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            auto &entry = _numericAccumulator[*maskIt];

            if (entry.state == NumericAccumulatorT::INITIALIZED) {
                *(currColId++) = *maskIt;
                *(currValue++) = entry.value;
            }

            _numericAccumulator.clear(*maskIt);
        }
    }

};

template<class IT, class NT>
class MaskedSPA2A {
private:
    using SymbolicAccumulator = SparseAccumulator2A<IT, void>;
    using NumericAccumulatorT = SparseAccumulator2A<IT, NT>;
    SymbolicAccumulator _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;

    explicit MaskedSPA2A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM)
            : _symbolicAccumulator(maxIndex), _numericAccumulator(maxIndex) {}

    [[nodiscard]]  std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    [[nodiscard]] SymbolicAccumulator &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] NumericAccumulatorT &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
//        assert(_symbolicAccumulator.isInitialized());

        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _symbolicAccumulator.setAllowed(*maskIt);
        }

        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                bool erased = _symbolicAccumulator.erase(B.colids[k]);
                if (erased) {
                    currRowNvals++;
                }
            }
        }

        // Reset - Remove all mask elements from the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _symbolicAccumulator.clear(*maskIt);
        }

        rowNvals[row] = currRowNvals;

//        assert(_symbolicAccumulator.isInitialized());
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            _numericAccumulator.setAllowed(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
//                auto &state = _numericAccumulator.getState(B.colids[k]);
//                if (state == NumericAccumulatorT::EMPTY) { continue; }
//
//                auto &value = _numericAccumulator.getValue(B.colids[k]);
//
//                if (state == NumericAccumulatorT::ALLOWED) {
//                    value = multop(A.values[j], B.values[k]);
//                    state = NumericAccumulatorT::INITIALIZED;
//                } else {
//                    assert(state == NumericAccumulatorT::INITIALIZED);
//                    value = addop(value, multop(A.values[j], B.values[k]));
//                }

                auto key = B.colids[k];
                if (_numericAccumulator._storage._states[key] == NumericAccumulatorT::EMPTY) { continue; }


                if (_numericAccumulator._storage._states[key] == NumericAccumulatorT::ALLOWED) {
                    _numericAccumulator._storage._values[key] = multop(A.values[j], B.values[k]);
                    _numericAccumulator._storage._states[key] = NumericAccumulatorT::INITIALIZED;
                } else {
//                    assert(state == NumericAccumulatorT::INITIALIZED);
                    _numericAccumulator._storage._values[key] = addop(_numericAccumulator._storage._values[key], multop(A.values[j], B.values[k]));
                }
            }
        }

        // Copy the values from the hash table to the output
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            auto &state = _numericAccumulator.getState(*maskIt);

            if (state == NumericAccumulatorT::INITIALIZED) {
                *(currColId++) = *maskIt;
                *(currValue++) = _numericAccumulator.getValue(*maskIt);
            }

            _numericAccumulator.clear(*maskIt);
        }
    }

};

#endif //MASKED_SPGEMM_MASKED_SPA_H
