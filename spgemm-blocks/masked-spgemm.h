#ifndef SPGEMM_GENERIC_H
#define SPGEMM_GENERIC_H

#include <algorithm>
#include "../CSR.h"
#include "scan.h"
#include "util.h"
#include "hash/HashAccumulator.h"
#include "hash/masked-hash.h"

template<template<class, class> class RowAlgorithm, class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM1p(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT>;
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = mallocAligned<IT>(A.rows);
    IT flops = calculateFlops(A, B, flopsPerRow);

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
        auto[upperBoundSizeC, maxRowSizeUpperBoundC, maxRowSizeA, maxRowSizeM]
        = scanInputs<true, RowAlg::CALC_MAX_ROW_UPPER_BOUND_SIZE_C, RowAlg::CALC_MAX_ROW_SIZE_A,
                RowAlg::CALC_MAX_ROW_SIZE_M>(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithm
        RowAlg alg;
        auto[bufferSize, bufferAlignment] = alg.getMemoryRequirement(maxRowSizeUpperBoundC, maxRowSizeA, maxRowSizeM);
        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        // Allocate temporary memory for C's column IDs and Values
        IT *colIdsLocal = my_malloc<IT>(upperBoundSizeC);
        NT *valuesLocal = my_malloc<NT>(upperBoundSizeC);
        IT *currColId = colIdsLocal;
        NT *currValue = valuesLocal;

        // Numeric phase
        alg.startNumeric(buffer, bufferSize, dirty);
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            alg.numericRow(A, B, M, multop, addop, row, currColId, currValue);
        }
        threadsNvals[thisThread] = currColId - colIdsLocal;
        alg.stopNumeric(dirty);

#pragma omp barrier
#pragma omp master
        {
            initC(A, B, C, threadsNvals, numThreads);
        }
#pragma omp barrier
        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);
        copyValuesToC(C, rowBeginIdx, colIdsLocal, valuesLocal, threadsNvals[thisThread]);
    }
}

template<template<class, class> class RowAlgorithm, class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM2p(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT>;
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = mallocAligned<IT>(A.rows);
    IT flops = calculateFlops(A, B, flopsPerRow);

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
        auto[upperBoundSizeC, maxRowSizeUpperBoundC, maxRowSizeA, maxRowSizeM]
        = scanInputs<false, RowAlg::CALC_MAX_ROW_UPPER_BOUND_SIZE_C, RowAlg::CALC_MAX_ROW_SIZE_A,
                RowAlg::CALC_MAX_ROW_SIZE_M>(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithm
        RowAlg alg;
        auto[bufferSize, bufferAlignment] = alg.getMemoryRequirement(maxRowSizeUpperBoundC, maxRowSizeA, maxRowSizeM);
        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        // Symbolic phase
        alg.startSymbolic(buffer, bufferSize, dirty);
        IT nvals = 0;
        for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
            alg.symbolicRow(A, B, M, row, rowNvals);
            nvals += rowNvals[row];
        }
        threadsNvals[thisThread] = nvals;
        alg.stopSymbolic(dirty);

        // init C
#pragma omp barrier
#pragma omp master
        {
            initC(A, B, C, threadsNvals, numThreads);
        }
#pragma omp barrier
        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);

        // Numeric phase
        alg.startNumeric(buffer, bufferSize, dirty);
        IT *currColId = &C.colids[C.rowptr[rowBeginIdx]];
        NT *currValue = &C.values[C.rowptr[rowBeginIdx]];
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (rowNvals[row] == 0) { continue; }
            alg.numericRow(A, B, M, multop, addop, row, currColId, currValue);
        }
        alg.stopNumeric(dirty);
    }
}

// region masked hash spgemm

template<class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedHashSpGEMM2p(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                        MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = mallocAligned<IT>(A.rows);
    IT flops = calculateFlops(A, B, flopsPerRow);

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

        // Symbolic phase
        {
            // Allocate auxiliary memory for the symbolic phase algorithm
            auto symAuxMemory{SymbolicMaskedHashAllocateAuxiliaryMemory(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M)};

            // Perform the row algorithm for each row
            IT nvals = 0;
            for (IT i = rowBeginIdx; i < rowEndIdx; i++) {
                SymbolicMaskedHashRow(A, B, M, i, rowNvals, symAuxMemory);
                nvals += rowNvals[i];
            }
            threadsNvals[thisThread] = nvals;
        }

        // init C
#pragma omp barrier
#pragma omp master
        {
            initC(A, B, C, threadsNvals, numThreads);
            std::cout << C.nnz << std::endl;
        }
#pragma omp barrier
        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);

        // Numeric phase
        {
            // Allocate auxiliary memory for the numeric phase algorithm
            auto auxMemory{NumericMaskedHashAllocateAuxiliaryMemory(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M)};

            // Perform the row algorithm for each row
            IT *currColId = &C.colids[C.rowptr[rowBeginIdx]];
            NT *currValue = &C.values[C.rowptr[rowBeginIdx]];

            for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
                NumericMaskedHashRow(A, B, M, multop, addop, i, currColId, currValue, auxMemory);
            }
        }
    }
}

template<class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedHashSpGEMM(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                      MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = my_malloc<IT>(A.rows);
    IT flops = calculateFlops(A, B, flopsPerRow);

    // Estimate row offsets
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

        // Estimate result size and allocate memory for C's column IDs and Values
        IT resultSize = estimateResultSize(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);
        IT *colIdsLocal = my_malloc<IT>(resultSize);
        NT *valuesLocal = my_malloc<NT>(resultSize);
        IT *currColId = colIdsLocal;
        NT *currValue = valuesLocal;

        // Allocate auxiliary memory for the algorithm
        auto auxMemory{NumericMaskedHashAllocateAuxiliaryMemory(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M)};

        // Perform the row algorithm for each row
        for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
            IT *currColIdOld = currColId;
            NumericMaskedHashRow(A, B, M, multop, addop, i, currColId, currValue, auxMemory);
            rowNvals[i] = currColId - currColIdOld;
        }
        threadsNvals[thisThread] = currColId - colIdsLocal;
        // TODO: free aux

#pragma omp barrier
#pragma omp master
        {
            initC(A, B, C, threadsNvals, numThreads);
        }
#pragma omp barrier
        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);
        copyValuesToC(C, rowBeginIdx, colIdsLocal, valuesLocal, threadsNvals[thisThread]);
    }
}

// endregion

#endif //SPGEMM_GENERIC_H
