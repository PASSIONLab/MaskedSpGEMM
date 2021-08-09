#ifndef MASKED_SPGEMM_MSA_ALGORITHM_H
#define MASKED_SPGEMM_MSA_ALGORITHM_H

namespace internal {
//region MSA Base

template<template<class, class, bool> class AccumT, class IT, class NT, bool Complemented>
class MSABase;

template<template<class, class, bool> class AccumT, class IT, class NT>
class MSABase<AccumT, IT, NT, false> {
public:
    inline const static bool COMPLEMENTED = false;
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;
    inline const static bool CALC_MAX_ROW_FLOPS = false;

protected:
    using SymbolicAccumulatorT = AccumT<IT, void, false>;
    using NumericAccumulatorT = AccumT<IT, NT, false>;
    SymbolicAccumulatorT _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    MSABase(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops)
            : _symbolicAccumulator(maxIndex), _numericAccumulator(maxIndex) {}

    [[nodiscard]]  std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    [[nodiscard]] SymbolicAccumulatorT &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] NumericAccumulatorT &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                     IT row, IT *rowNvals, IT &flops) {
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
public:
    inline const static bool COMPLEMENTED = true;
    inline const static bool CALC_MAX_ROW_SIZE_A = false;
    inline const static bool CALC_MAX_ROW_SIZE_M = false;
    inline const static bool CALC_MAX_ROW_FLOPS = false;

protected:
    using SymbolicAccumulatorT = AccumT<IT, void, true>;
    using NumericAccumulatorT = AccumT<IT, NT, true>;
    SymbolicAccumulatorT _symbolicAccumulator;
    NumericAccumulatorT _numericAccumulator;

public:
    MSABase(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops)
            : _symbolicAccumulator(maxIndex), _numericAccumulator(maxIndex) {}

    [[nodiscard]]  std::tuple<size_t, size_t> getMemoryRequirement() {
        auto[symbolicSize, symbolicAlignment] = _symbolicAccumulator.getMemoryRequirement();
        auto[numericSize, numericAlignment] = _numericAccumulator.getMemoryRequirement();

        return {std::max(symbolicSize, numericSize), std::lcm(symbolicAlignment, numericAlignment)};
    }

    [[nodiscard]] SymbolicAccumulatorT &getSymbolicAccumulator() { return _symbolicAccumulator; }

    [[nodiscard]] NumericAccumulatorT &getNumericAccumulator() { return _numericAccumulator; }

    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                     IT row, IT *rowNvals, IT &flops) {
//        assert(_symbolicAccumulator.isInitialized());

        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        IT currRowNvals = 0;
        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                if (_symbolicAccumulator.getState(B.colids[k]) != SymbolicAccumulatorT::INITIALIZED) {
                    _symbolicAccumulator.setInitialized(B.colids[k]);
                    currRowNvals++;
//                    std::cout << "I " << B.colids[k] << std::endl;
                }
            }
        }

        // Remove all mask elements from the accumulator
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            bool erased = _symbolicAccumulator.erase(*maskIt);
            if (erased) { currRowNvals--; }
//            if (erased) { std::cout << "E " << *maskIt << std::endl; }
        }

//        std::cout << std::endl;

        _symbolicAccumulator.resetStates();

        rowNvals[row] = currRowNvals;

//        assert(_symbolicAccumulator.isInitialized());
    }
};

//endregion

//region MSA 1A

template<class IT, class NT, bool Complemented, bool Sorted>
class MSA1A;

template<class IT, class NT>
class MSA1A<IT, NT, false, false> : public MSABase<MaskedSparseAccumulator1A, IT, NT, false> {
    using super = MSABase<MaskedSparseAccumulator1A, IT, NT, false>;

public:
    explicit MSA1A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops)
            : super(maxIndex, maxRowSizeA, maxRowSizeM, maxRowFlops) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue, IT &flops) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            this->_numericAccumulator.setAllowed(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                auto &entry = this->_numericAccumulator[B.colids[k]];

                if (entry.state == super::NumericAccumulatorT::ALLOWED) {
                    entry.value = multop(A.values[j], B.values[k]);
                    entry.state = super::NumericAccumulatorT::INITIALIZED;
                } else if (entry.state == super::NumericAccumulatorT::INITIALIZED) {
                    entry.value = addop(entry.value, multop(A.values[j], B.values[k]));
                }
            }
        }

        // Copy the values from the hash table to the output
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            auto &entry = this->_numericAccumulator[*maskIt];

            if (entry.state == super::NumericAccumulatorT::INITIALIZED) {
                *(currColId++) = *maskIt;
                *(currValue++) = entry.value;
            }

            this->_numericAccumulator.clear(*maskIt);
        }
    }
};

template<class IT, class NT, bool Sorted>
class MSA1A<IT, NT, true, Sorted> : public MSABase<MaskedSparseAccumulator1A, IT, NT, true> {
    using super = MSABase<MaskedSparseAccumulator1A, IT, NT, true>;

public:
    explicit MSA1A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops)
            : super(maxIndex, maxRowSizeA, maxRowSizeM, maxRowFlops) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue, IT &flops) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            this->_numericAccumulator.setNotAllowed(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                auto &entry = this->_numericAccumulator[B.colids[k]];

                if (entry.state == super::NumericAccumulatorT::ALLOWED) {
                    entry.value = multop(A.values[j], B.values[k]);
                    this->_numericAccumulator.setInitialized(B.colids[k]);
                } else if (entry.state == super::NumericAccumulatorT::INITIALIZED) {
                    entry.value = addop(entry.value, multop(A.values[j], B.values[k]));
                }
            }
        }

        this->_numericAccumulator.template gather<Sorted>(currColId, currValue);
        this->_numericAccumulator.resetStates(maskBegin, maskEnd);
    }
};

//endregion

//region MSA 2A

template<class IT, class NT, bool Complemented, bool Sorted>
class MSA2A;

template<class IT, class NT>
class MSA2A<IT, NT, false, false> : public MSABase<MaskedSparseAccumulator2A, IT, NT, false> {
    using super = MSABase<MaskedSparseAccumulator2A, IT, NT, false>;

public:
    explicit MSA2A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops)
            : super(maxIndex, maxRowSizeA, maxRowSizeM, maxRowFlops) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue, IT &flops) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            this->_numericAccumulator.setAllowed(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                auto &state = this->_numericAccumulator.getState(B.colids[k]);
                if (state == super::NumericAccumulatorT::NOT_ALLOWED) { continue; }

                auto &value = this->_numericAccumulator.getValue(B.colids[k]);

                if (state == super::NumericAccumulatorT::ALLOWED) {
                    value = multop(A.values[j], B.values[k]);
                    state = super::NumericAccumulatorT::INITIALIZED;
                } else {
                    assert(state == super::NumericAccumulatorT::INITIALIZED);
                    value = addop(value, multop(A.values[j], B.values[k]));
                }
            }
        }

        // Copy the values from the hash table to the output
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            auto &state = this->_numericAccumulator.getState(*maskIt);

            if (state == super::NumericAccumulatorT::INITIALIZED) {
                *(currColId++) = *maskIt;
                *(currValue++) = this->_numericAccumulator.getValue(*maskIt);
            }

            this->_numericAccumulator.clear(*maskIt);
        }
    }
};

template<class IT, class NT, bool Sorted>
class MSA2A<IT, NT, true, Sorted> : public MSABase<MaskedSparseAccumulator2A, IT, NT, true> {
    using super = MSABase<MaskedSparseAccumulator2A, IT, NT, true>;

public:
    explicit MSA2A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM, IT maxRowFlops)
            : super(maxIndex, maxRowSizeA, maxRowSizeM, maxRowFlops) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue, IT &flops) {
        const auto maskBegin = &M.colids[M.rowptr[row]];
        const auto maskEnd = &M.colids[M.rowptr[row + 1]];

        // Insert all mask elements to the SPA
        for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
            this->_numericAccumulator.setNotAllowed(*maskIt);
        }

        for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
            IT inner = A.colids[j];
            for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                auto &state = this->_numericAccumulator.getState(B.colids[k]);
                if (state == super::NumericAccumulatorT::NOT_ALLOWED) { continue; }

                auto &value = this->_numericAccumulator.getValue(B.colids[k]);

                if (state == super::NumericAccumulatorT::ALLOWED) {
                    value = multop(A.values[j], B.values[k]);
                    this->_numericAccumulator.setInitialized(B.colids[k]);
                } else {
                    assert(state == super::NumericAccumulatorT::INITIALIZED);
                    value = addop(value, multop(A.values[j], B.values[k]));
                }
            }
        }

        this->_numericAccumulator.template gather<Sorted>(currColId, currValue);
        this->_numericAccumulator.resetStates(maskBegin, maskEnd);
    }
};

//endregion
}

enum class MSAType {
    OneArray,
    TwoArrays,
};

template<bool Complemented, bool Sorted, MSAType NumArrays>
struct MSA {
    template<class IT, class NT>
    using Impl = std::conditional_t<NumArrays == MSAType::OneArray, internal::MSA1A<IT, NT, Complemented, Sorted>,
            std::conditional_t<NumArrays == MSAType::TwoArrays, internal::MSA2A<IT, NT, Complemented, Sorted>,
                    void>>;
};

template<bool Complemented, bool Sorted>
struct MSA1A {
    template<class IT, class NT>
    using Impl = internal::MSA1A<IT, NT, Complemented, Sorted>;
};

template<bool Complemented, bool Sorted>
struct MSA2A {
    template<class IT, class NT>
    using Impl = internal::MSA2A<IT, NT, Complemented, Sorted>;
};


#endif //MASKED_SPGEMM_MSA_ALGORITHM_H
