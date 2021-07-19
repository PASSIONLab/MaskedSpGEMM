#ifndef MASKED_SPGEMM_MASKED_INNER_SPGEMM_H
#define MASKED_SPGEMM_MASKED_INNER_SPGEMM_H

#include "../../CSR.h"
#include "../../CSC.h"
#include "../common.h"
#include "../scan.h"

/*
 * Manually inlined numeric and symbolic funcs
 */

template<class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM1pInnerProduct(const CSR<IT, NT> &A, const CSC<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *workPerRow = my_malloc<IT>(M.rows, false);
    IT work = calculateWork(A, B, M, workPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(M.rows, false);
    exclusiveScan(workPerRow, M.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(M.rows, false);
    IT *threadsNvals = my_malloc<IT>(numThreads, false);

    // Allocate temporary memory for C's column IDs and values
    IT *colIds = my_malloc<IT>(M.nnz, false);
    NT *values = my_malloc<NT>(M.nnz, false);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();

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
            for (IT j = M.rowptr[row]; j < M.rowptr[row + 1]; ++j) {
                IT itAIdx = A.rowptr[row];
                const IT rowAEnd = A.rowptr[row + 1];
                if (itAIdx == rowAEnd) { continue; }

                IT itBIdx = B.colptr[M.colids[j]];
                const IT colBEnd = B.colptr[M.colids[j] + 1];
                if (itBIdx == colBEnd) { continue; }

                bool active = false;
                NT value;

                while (true) {
                    if (A.colids[itAIdx] < B.rowids[itBIdx]) {
                        if (++itAIdx == rowAEnd) { break; }
                    } else if (A.colids[itAIdx] > B.rowids[itBIdx]) {
                        if (++itBIdx == colBEnd) { break; }
                    } else {
                        if (active) {
                            value = addop(value, multop(A.values[itAIdx], B.values[itBIdx]));
                        } else {
                            active = true;
                            value = multop(A.values[itAIdx], B.values[itBIdx]);
                        }
                        if (++itAIdx == rowAEnd) { break; }
                        if (++itBIdx == colBEnd) { break; }
                    }
                }

                if (active) {
                    *(currColId++) = M.colids[j];
                    *(currValue++) = value;
                }
            }
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

template<class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM2pInnerProduct(const CSR<IT, NT> &A, const CSC<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {
    // Calculate number of threads and init C
    setNumThreads(numThreads);
    verifyInputs(A, B, C, M);

    // Estimate work
    IT *workPerRow = my_malloc<IT>(M.rows, false);
    IT work = calculateWork(A, B, M, workPerRow, numThreads);

    // Calculate cumulative work
    IT *cumulativeWork = my_malloc<IT>(M.rows, false);
    exclusiveScan(workPerRow, M.rows, cumulativeWork, numThreads);

    // Allocate memory for row sizes
    IT *rowNvals = my_malloc<IT>(M.rows, false);
    IT *threadsNvals = my_malloc<IT>(numThreads, false);

#pragma omp parallel num_threads(numThreads)
    {
        int thisThread = omp_get_thread_num();

        // Distribute work
        auto[rowBeginIdx, rowEndIdx] = distributeWork(work, cumulativeWork, A.rows, numThreads, thisThread);

        // Symbolic phase
        IT nvals = 0;
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (workPerRow[row] == 0) {
                rowNvals[row] = 0;
                continue;
            }

            IT currRowNvals = 0;

            for (IT j = M.rowptr[row]; j < M.rowptr[row + 1]; ++j) {
                IT itAIdx = A.rowptr[row];
                const IT rowAEnd = A.rowptr[row + 1];
                if (itAIdx == rowAEnd) { continue; }

                IT itBIdx = B.colptr[M.colids[j]];
                const IT colBEnd = B.colptr[M.colids[j] + 1];
                if (itBIdx == colBEnd) { continue; }

                while (true) {
                    if (A.colids[itAIdx] < B.rowids[itBIdx]) {
                        if (++itAIdx == rowAEnd) { break; }
                    } else if (A.colids[itAIdx] > B.rowids[itBIdx]) {
                        if (++itBIdx == colBEnd) { break; }
                    } else {
                        currRowNvals++;
                        break;
                    }
                }
            }

            rowNvals[row] = currRowNvals;
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
            for (IT j = M.rowptr[row]; j < M.rowptr[row + 1]; ++j) {
                IT itAIdx = A.rowptr[row];
                const IT rowAEnd = A.rowptr[row + 1];
                if (itAIdx == rowAEnd) { continue; }

                IT itBIdx = B.colptr[M.colids[j]];
                const IT colBEnd = B.colptr[M.colids[j] + 1];
                if (itBIdx == colBEnd) { continue; }

                bool active = false;
                NT value;

                while (true) {
                    if (A.colids[itAIdx] < B.rowids[itBIdx]) {
                        if (++itAIdx == rowAEnd) { break; }
                    } else if (A.colids[itAIdx] > B.rowids[itBIdx]) {
                        if (++itBIdx == colBEnd) { break; }
                    } else {
                        if (active) {
                            value = addop(value, multop(A.values[itAIdx], B.values[itBIdx]));
                        } else {
                            active = true;
                            value = multop(A.values[itAIdx], B.values[itBIdx]);
                        }
                        if (++itAIdx == rowAEnd) { break; }
                        if (++itBIdx == colBEnd) { break; }
                    }
                }

                if (active) {
                    *(currColId++) = M.colids[j];
                    *(currValue++) = value;
                }
            }
            rowNvals[row] = currColId - rowColIdBegin;
        }
    }

    my_free(workPerRow, cumulativeWork, rowNvals, threadsNvals);
}


#endif //MASKED_SPGEMM_MASKED_INNER_SPGEMM_H
