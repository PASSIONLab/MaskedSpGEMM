#ifndef MASKED_SPGEMM_MASKED_SPGEMM_INNER_H
#define MASKED_SPGEMM_MASKED_SPGEMM_INNER_H

#include "InnerAlgorithm.h"

template<template<class, class, bool> class RowAlgorithm, bool Complemented = false,
        class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM1p(const CSR<IT, NT> &A, const CSC<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT, Complemented>;

    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *workPerRow = my_malloc<IT>(M.rows);
    IT work = calculateWork(A, B, M, workPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(M.rows);
    exclusiveScan(workPerRow, M.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(M.rows);
    IT *threadsNvals = my_malloc<IT>(numThreads);

    // Allocate temporary memory for C's column IDs and values
    IT *colIds = my_malloc<IT>(M.nnz);
    NT *values = my_malloc<NT>(M.nnz);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();
        RowAlg alg;

        // Distribute work
        auto[rowBeginIdx, rowEndIdx] = distributeWork(work, cumulativeWork, A.rows, numThreads, thisThread);

        // Get arrays for local colIDs and values
        IT *const colIdsLocal = colIds + M.rowptr[rowBeginIdx];
        NT *const valuesLocal = values + M.rowptr[rowBeginIdx];
        IT *currColId = colIdsLocal;
        NT *currValue = valuesLocal;

        // Numeric phase
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (workPerRow[row] == 0) {
                rowNvals[row] = 0;
            }

            auto rowColIdBegin = currColId;
            alg.numericRow(A, B, M, multop, addop, row, currColId, currValue);
            rowNvals[row] = currColId - rowColIdBegin;
        }

        threadsNvals[thisThread] = currColId - colIdsLocal;

#pragma omp barrier
#pragma omp master
        {
            initC(A, B, C, threadsNvals, numThreads);
        }
#pragma omp barrier

        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);
        copyValuesToC(C, rowBeginIdx, colIdsLocal, valuesLocal, threadsNvals[thisThread]);
    }

    my_free(workPerRow, cumulativeWork, rowNvals, threadsNvals, colIds, values);
}

template<template<class, class, bool> class RowAlgorithm, bool Complemented = false,
        class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM2p(const CSR<IT, NT> &A, const CSC<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    using RowAlg = RowAlgorithm<IT, NT, Complemented>;

    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *workPerRow = my_malloc<IT>(M.rows);
    IT work = calculateWork(A, B, M, workPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(M.rows);
    exclusiveScan(workPerRow, M.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(M.rows);
    IT *threadsNvals = my_malloc<IT>(numThreads);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();
        RowAlg alg;

        // Distribute work
        auto[rowBeginIdx, rowEndIdx] = distributeWork(work, cumulativeWork, A.rows, numThreads, thisThread);

        // Symbolic phase
        IT nvals = 0;
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (workPerRow[row] == 0) {
                rowNvals[row] = 0;
                continue;
            }

            alg.symbolicRow(A, B, M, row, rowNvals);
            nvals += rowNvals[row];
        }
        threadsNvals[thisThread] = nvals;

        // init C
#pragma omp barrier
#pragma omp master
        {
            initC(A, B, C, threadsNvals, numThreads);
        }
#pragma omp barrier
        setRowOffsets(C, threadsNvals, rowBeginIdx, rowEndIdx, rowNvals, numThreads, thisThread);

        // Numeric phase
        IT *currColId = &C.colids[C.rowptr[rowBeginIdx]];
        NT *currValue = &C.values[C.rowptr[rowBeginIdx]];
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (workPerRow[row] == 0) {
                rowNvals[row] = 0;
                continue;
            }

            auto rowColIdBegin = currColId;
            alg.numericRow(A, B, M, multop, addop, row, currColId, currValue);
            rowNvals[row] = currColId - rowColIdBegin;
        }
    }

    my_free(workPerRow, cumulativeWork, rowNvals, threadsNvals);
}

#endif //MASKED_SPGEMM_MASKED_SPGEMM_INNER_H
