#ifndef MASKED_SPGEMM_MSA_ALGORITHM_H
#define MASKED_SPGEMM_MSA_ALGORITHM_H

#include "MSAAlgorithmBase.h"

//region MSA 1A

template<class IT, class NT, bool Complemented>
class MSA1A;

template<class IT, class NT>
class MSA1A<IT, NT, false> : public MSABase<MaskedSparseAccumulator1A, IT, NT, false> {
    using super = MSABase<MaskedSparseAccumulator1A, IT, NT, false>;

public:
    explicit MSA1A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) : super(maxIndex, maxRowSizeA, maxRowSizeM) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
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

template<class IT, class NT>
class MSA1A<IT, NT, true> : public MSABase<MaskedSparseAccumulator1A, IT, NT, true> {
    using super = MSABase<MaskedSparseAccumulator1A, IT, NT, true>;

public:
    explicit MSA1A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) : super(maxIndex, maxRowSizeA, maxRowSizeM) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
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

        this->_numericAccumulator.gather(currColId, currValue);
        this->_numericAccumulator.resetStates(maskBegin, maskEnd);
    }
};

//endregion

//region MSA 2A

template<class IT, class NT, bool Complemented>
class MSA2A;

template<class IT, class NT>
class MSA2A<IT, NT, false> : public MSABase<MaskedSparseAccumulator2A, IT, NT, false> {
    using super = MSABase<MaskedSparseAccumulator2A, IT, NT, false>;

public:
    explicit MSA2A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) : super(maxIndex, maxRowSizeA, maxRowSizeM) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
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

            this-> _numericAccumulator.clear(*maskIt);
        }
    }
};

template<class IT, class NT>
class MSA2A<IT, NT, true> : public MSABase<MaskedSparseAccumulator2A, IT, NT, true> {
    using super = MSABase<MaskedSparseAccumulator2A, IT, NT, true>;

public:
    explicit MSA2A(IT maxIndex, IT maxRowSizeA, IT maxRowSizeM) : super(maxIndex, maxRowSizeA, maxRowSizeM) {}

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
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

        this->_numericAccumulator.gather(currColId, currValue);
        this->_numericAccumulator.resetStates(maskBegin, maskEnd);
    }
};

//endregion

#endif //MASKED_SPGEMM_MSA_ALGORITHM_H
