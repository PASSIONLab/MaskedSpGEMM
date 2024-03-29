#ifndef SPGEMM_GENERIC_H
#define SPGEMM_GENERIC_H

#include <algorithm>
#include "../CSR.h"
#include "scan.h"
#include "util.h"
#include "hash/HashAccumulator.h"
#include "hash/MaskedHashAlgorithm.h"
#include "msa/MSAAlgorithm_old.h"
#include "msa/MSAAlgorithm.h"
#include "mca/MCAAlgorithm.h"
#include "heap/MaskedHeapAlgorithm.h"
#include "heap/MaskedHeapAlgorithmInspect.h"


template<template<class, class> class RowAlgorithm, class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM1p(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT>;
    const bool Complemented = RowAlg::COMPLEMENTED;

    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = my_malloc<IT>(A.rows, false);
    IT flops = Complemented ? calculateFlops(A, B, flopsPerRow, numThreads)
                            : calculateFlops(A, B, M, flopsPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(A.rows, false);
    exclusiveScan(flopsPerRow, A.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(A.rows, false);
    IT *threadsNvals = my_malloc<IT>(numThreads, false);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();

        // Distribute work
        auto[rowBeginIdx, rowEndIdx] = distributeWork(flops, cumulativeWork, A.rows, numThreads, thisThread);

        // Scan the input matrices
        auto[upperBoundSizeC, maxRowSizeA, maxRowSizeM, maxRowFlops]
        = scanInputs<Complemented,
                true,
                RowAlg::CALC_MAX_ROW_SIZE_A,
                RowAlg::CALC_MAX_ROW_SIZE_M,
                RowAlg::CALC_MAX_ROW_FLOPS>(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithm
        RowAlg alg{B.cols, maxRowSizeA, maxRowSizeM, maxRowFlops};
        auto[bufferSize, bufferAlignment] = alg.getMemoryRequirement();
        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        // Allocate temporary memory for C's column IDs and Values
        IT *colIdsLocal = my_malloc<IT>(upperBoundSizeC, false);
        NT *valuesLocal = my_malloc<NT>(upperBoundSizeC, false);
        IT *currColId = colIdsLocal;
        NT *currValue = valuesLocal;

        // Numeric phase
        alg.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (flopsPerRow[row]) {
                auto rowColIdBegin = currColId;
                alg.numericRow(A, B, M, multop, addop, row, currColId, currValue, flopsPerRow[row]);
                rowNvals[row] = currColId - rowColIdBegin;
            } else {
                rowNvals[row] = 0;
            }
        }
        threadsNvals[thisThread] = currColId - colIdsLocal;
        dirty = alg.getNumericAccumulator().releaseBuffer();

#pragma omp barrier
#pragma omp master
        {
            initC(A, B, C, threadsNvals, numThreads);
        }
#pragma omp barrier
        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);
        copyValuesToC(C, rowBeginIdx, colIdsLocal, valuesLocal, threadsNvals[thisThread]);

        my_free(colIdsLocal, valuesLocal);
        freeAligned(buffer);
    }

    my_free(flopsPerRow, cumulativeWork, rowNvals, threadsNvals);
}

template<template<class, class> class RowAlgorithm, class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM2p(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT>;
    const bool Complemented = RowAlg::COMPLEMENTED;

    CSR<IT, NT> R;

    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = my_malloc<IT>(A.rows, false);
    IT flops = Complemented ? calculateFlops(A, B, flopsPerRow, numThreads)
                            : calculateFlops(A, B, M, flopsPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(A.rows, false);
    exclusiveScan(flopsPerRow, A.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(A.rows, false);
    IT *threadsNvals = my_malloc<IT>(numThreads, false);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();

        // Distribute work
        auto[rowBeginIdx, rowEndIdx] = distributeWork(flops, cumulativeWork, A.rows, numThreads, thisThread);

        // Scan the input matrices
        auto[upperBoundSizeC, maxRowSizeA, maxRowSizeM, maxRowFlops]
        = scanInputs<Complemented,
                false,
                RowAlg::CALC_MAX_ROW_SIZE_A,
                RowAlg::CALC_MAX_ROW_SIZE_M,
                RowAlg::CALC_MAX_ROW_FLOPS>(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithm
        RowAlg alg{B.cols, maxRowSizeA, maxRowSizeM, maxRowFlops};
        auto[bufferSize, bufferAlignment] = alg.getMemoryRequirement();
        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        // Symbolic phase
        alg.getSymbolicAccumulator().setBuffer(buffer, bufferSize, dirty);
        IT nvals = 0;
        for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
            if (flopsPerRow[row]) {
                alg.symbolicRow(A, B, M, row, rowNvals, flopsPerRow[row]);
            } else {
                rowNvals[row] = 0;
            }
            nvals += rowNvals[row];
        }
        threadsNvals[thisThread] = nvals;
        dirty = alg.getSymbolicAccumulator().releaseBuffer();

        // init C
#pragma omp barrier
#pragma omp master
        {
            initC(A, B, R, threadsNvals, numThreads);
        }
#pragma omp barrier
        setRowOffsets(R, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);

        // Numeric phase
        alg.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
        IT *currColId = &R.colids[R.rowptr[rowBeginIdx]];
        NT *currValue = &R.values[R.rowptr[rowBeginIdx]];
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (rowNvals[row] == 0) { continue; }
            alg.numericRow(A, B, M, multop, addop, row, currColId, currValue, rowNvals[row]);
        }
        dirty = alg.getNumericAccumulator().releaseBuffer();

        freeAligned(buffer);
    }

    my_free(flopsPerRow, cumulativeWork, rowNvals, threadsNvals);

    // TODO: use move ctr
    C.make_empty();
    C.rows = R.rows;
    C.cols = R.cols;
    C.nnz = R.nnz;
    C.rowptr = R.rowptr;
    C.colids = R.colids;
    C.values = R.values;

    R.rows = 0;
    R.cols = 0;
    R.nnz = 0;
}

#endif //SPGEMM_GENERIC_H
