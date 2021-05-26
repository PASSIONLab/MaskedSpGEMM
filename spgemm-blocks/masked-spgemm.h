#ifndef SPGEMM_GENERIC_H
#define SPGEMM_GENERIC_H

#include <algorithm>
#include "../CSR.h"
#include "scan.h"
#include "util.h"
#include "hash/HashAccumulator.h"
#include "hash/MaskedHash.h"
#include "msa/MSA.h"
#include "mca/MCA.h"
#include "heap/MaskedHeap.h"


template<template<class, class> class RowAlgorithm, class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM1p(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT>;
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = my_malloc<IT>(A.rows);
    IT flops = calculateFlops(A, B, M, flopsPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(A.rows);
    exclusiveScan(flopsPerRow, A.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(A.rows);
    IT *threadsNvals = my_malloc<IT>(numThreads);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();

        // Distribute work
        auto[rowBeginIdx, rowEndIdx] = distributeWork(flops, cumulativeWork, A.rows, numThreads, thisThread);

        // Scan the input matrices
        auto[upperBoundSizeC, maxRowSizeA, maxRowSizeM] = scanInputs<true, RowAlg::CALC_MAX_ROW_SIZE_A, RowAlg::CALC_MAX_ROW_SIZE_M>(
                rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithm
        RowAlg alg{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSize, bufferAlignment] = alg.getMemoryRequirement();
        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        // Allocate temporary memory for C's column IDs and Values
        IT *colIdsLocal = my_malloc<IT>(upperBoundSizeC);
        NT *valuesLocal = my_malloc<NT>(upperBoundSizeC);
        IT *currColId = colIdsLocal;
        NT *currValue = valuesLocal;

        // Numeric phase
        alg.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (flopsPerRow[row]) {
                alg.numericRow(A, B, M, multop, addop, row, currColId, currValue);
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
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = my_malloc<IT>(A.rows);
    IT flops = calculateFlops(A, B, M, flopsPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(A.rows);
    exclusiveScan(flopsPerRow, A.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(A.rows);
    IT *threadsNvals = my_malloc<IT>(numThreads);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();

        // Distribute work
        auto[rowBeginIdx, rowEndIdx] = distributeWork(flops, cumulativeWork, A.rows, numThreads, thisThread);

        // Scan the input matrices
        auto[upperBoundSizeC, maxRowSizeA, maxRowSizeM] = scanInputs<false, RowAlg::CALC_MAX_ROW_SIZE_A, RowAlg::CALC_MAX_ROW_SIZE_M>(
                rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithm
        RowAlg alg{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSize, bufferAlignment] = alg.getMemoryRequirement();
        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        // Symbolic phase
        alg.getSymbolicAccumulator().setBuffer(buffer, bufferSize, dirty);
        IT nvals = 0;
        for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
            if (flopsPerRow[row]) {
                alg.symbolicRow(A, B, M, row, rowNvals);
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
            initC(A, B, C, threadsNvals, numThreads);
        }
#pragma omp barrier
        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);

        // Numeric phase
        alg.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
        IT *currColId = &C.colids[C.rowptr[rowBeginIdx]];
        NT *currValue = &C.values[C.rowptr[rowBeginIdx]];
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (rowNvals[row] == 0) { continue; }
            alg.numericRow(A, B, M, multop, addop, row, currColId, currValue);
        }
        dirty = alg.getNumericAccumulator().releaseBuffer();

        freeAligned(buffer);
    }

    my_free(flopsPerRow, cumulativeWork, rowNvals, threadsNvals);
}

#endif //SPGEMM_GENERIC_H
