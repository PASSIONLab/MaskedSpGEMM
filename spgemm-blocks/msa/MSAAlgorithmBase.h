#ifndef MASKED_SPGEMM_MSA_ALGORITHM_BASE_H
#define MASKED_SPGEMM_MSA_ALGORITHM_BASE_H

template<template<class, class, bool> class AccumT, class IT, class NT, bool Complemented>
class MSABase;

template<template<class, class, bool> class AccumT, class IT, class NT>
class MSABase<AccumT, IT, NT, false> {
protected:
    using SymbolicAccumulator = AccumT<IT, void, false>;
    using NumericAccumulatorT = AccumT<IT, NT, false>;
    SymbolicAccumulator _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;

    explicit MSABase(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM)
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
};

template<template<class, class, bool> class AccumT, class IT, class NT>
class MSABase<AccumT, IT, NT, true> {
protected:
    using SymbolicAccumulator = AccumT<IT, void, true>;
    using NumericAccumulatorT = AccumT<IT, NT, true>;
    SymbolicAccumulator _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;

    explicit MSABase(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM)
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

        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                if (_symbolicAccumulator.getState(B.colids[k]) != SymbolicAccumulator::INITIALIZED) {
                    _symbolicAccumulator.setInitialized(B.colids[k]);
                    currRowNvals++;
                }
            }
        }

        // Remove all mask elements from the accumulator
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            bool erased = _symbolicAccumulator.erase(*maskIt);
            if (erased) { currRowNvals--; }
        }

        _symbolicAccumulator.resetStates();

        rowNvals[row] = currRowNvals;

//        assert(_symbolicAccumulator.isInitialized());
    }
};

#endif //MASKEDSPGEMM_MSAALGORITHMBASE_H
