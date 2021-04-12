
#ifndef MASKEDSPGEMM_MASKED_SPGEMM_PROF_H
#define MASKEDSPGEMM_MASKED_SPGEMM_PROF_H

#include <x86intrin.h>

template<template<class, class> class RowAlgorithm, class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM1p_prof(long *rowTimes, const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT>;
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *flopsPerRow = my_malloc<IT>(A.rows);
    IT flops = calculateFlops(A, B, flopsPerRow, numThreads);

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

        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> end;

        // Numeric phase
        alg.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
        start = std::chrono::high_resolution_clock::now();
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (M.rowptr[row] != M.rowptr[row + 1] && A.rowptr[row] != A.rowptr[row + 1]) {
                alg.numericRow(A, B, M, multop, addop, row, currColId, currValue);
            } else {
                rowNvals[row] = 0;
            }
            end = std::chrono::high_resolution_clock::now();
            rowTimes[row] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            start = end;
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

        my_free(colIdsLocal);
        my_free(valuesLocal);
        freeAligned(buffer);
    }

    my_free(flopsPerRow);
    my_free(cumulativeWork);
    my_free(rowNvals);
    my_free(threadsNvals);
}

#endif //MASKEDSPGEMM_MASKED_SPGEMM_PROF_H
