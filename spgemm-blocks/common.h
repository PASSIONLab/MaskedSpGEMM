#ifndef MASKED_SPGEMM_COMMON_H
#define MASKED_SPGEMM_COMMON_H

void setNumThreads(unsigned &numThreads) {
    if (numThreads == 0) {
#pragma omp parallel
#pragma omp single
        numThreads = omp_get_num_threads();
    }
}

template<typename IT, typename NT>
void verifyInputs(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M) {
    assert(A.cols == B.rows);
    assert(M.rows == A.rows);
    assert(M.cols == B.cols);

    if (!C.isEmpty()) { C.make_empty(); }
}

template<typename IT, typename NT>
IT calculateFlops(const CSR<IT, NT> &A, const CSR<IT, NT> &B, IT *flopsPerRow) {
    IT flops = 0; // total flop (multiplication) needed to generate C

#pragma omp parallel for reduction(+:flops)
    for (IT i = 0; i < A.rows; ++i) {
        IT flopsRow = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
            IT inner = A.colids[j];
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];
            flopsRow += npins;
        }
        flopsPerRow[i] = flopsRow;
        flops += flopsRow;
    }

    return flops;
}


template<class IT>
std::tuple<IT, IT> distributeWork(IT totalWork, IT *accumWorkPerRow, IT nrows, int numThreads, int thisThread) {
    IT workPerThread = totalWork / numThreads;

    // @formatter:off
    IT rowBegin = thisThread != 0              ? (lower_bound(accumWorkPerRow, accumWorkPerRow + nrows, workPerThread * thisThread))       - accumWorkPerRow : 0;
    IT rowEnd   = thisThread != numThreads - 1 ? (lower_bound(accumWorkPerRow, accumWorkPerRow + nrows, workPerThread * (thisThread + 1))) - accumWorkPerRow : nrows;
    // @formatter:on

    return {rowBegin, rowEnd};
}

template<class IT, class NT>
IT estimateResultSize(IT rowBeginIdx, IT rowEndIdx, IT *flopsPerRow,
                      const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
    IT size = 0;
    for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
        size += std::min(flopsPerRow[row], M.rowptr[row + 1] - M.rowptr[row]);
    }
    return size;
}

template<bool calcUpperBoundSizeC, bool calcMaxRowUpperSizeBoundC, bool calcMaxRowSizeA, bool calcMaxRowSizeM,
        class IT, class NT>
std::tuple<IT, IT, IT, IT> scanInputs(IT rowBeginIdx, IT rowEndIdx, IT *flopsPerRow,
                                      const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
    IT sizeC = 0;
    IT maxRowSizeC = 0;
    IT maxRowSizeM = 0;
    IT maxRowSizeA = 0;

    for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
        if (calcUpperBoundSizeC || calcMaxRowUpperSizeBoundC) {
            IT rowSize = std::min(flopsPerRow[row], M.rowptr[row + 1] - M.rowptr[row]);
            if (calcUpperBoundSizeC) { sizeC += rowSize; }
            if (calcMaxRowUpperSizeBoundC) { maxRowSizeC = std::max(maxRowSizeC, rowSize); }
        }

        if (calcMaxRowSizeA) { maxRowSizeA = std::max(maxRowSizeA, A.rowptr[row + 1] - A.rowptr[row]); }
        if (calcMaxRowSizeM) { maxRowSizeM = std::max(maxRowSizeM, M.rowptr[row + 1] - M.rowptr[row]); }
    }

    return {sizeC, maxRowSizeC, maxRowSizeA, maxRowSizeM};
}

template<class IT, class NT>
void initC(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, IT *threadsNvals, int numThreads) {
    C.rows = A.rows;
    C.cols = B.cols;
    C.rowptr = my_malloc<IT>(C.rows + 1);

    C.nnz = std::accumulate(threadsNvals, threadsNvals + numThreads, IT(0));
    C.colids = my_malloc<IT>(C.nnz);
    C.values = my_malloc<NT>(C.nnz);
}

template<class IT, class NT>
void setRowOffsets(CSR<IT, NT> &C, IT *threadsNvals, IT rowBeginIdx, IT rowEndIdx, IT *rowNvals,
                   int numThreads, int thisThread) {
    IT rowPtrOffset = std::accumulate(threadsNvals, threadsNvals + thisThread, IT(0));

    // set rowptr in C for local rows
    for (IT i = rowBeginIdx; i < rowEndIdx; ++i) {
        C.rowptr[i] = rowPtrOffset;
        rowPtrOffset += rowNvals[i];
    }
    if (thisThread == numThreads - 1) { C.rowptr[C.rows] = rowPtrOffset; }
}

template<class IT, class NT>
void copyValuesToC(CSR<IT, NT> &C, IT rowBeginIdx, IT *colIdsLocal, NT *valuesLocal, IT threadNvals) {
    copy(colIdsLocal, colIdsLocal + threadNvals, C.colids + C.rowptr[rowBeginIdx]);
    copy(valuesLocal, valuesLocal + threadNvals, C.values + C.rowptr[rowBeginIdx]);
}

#endif //MASKED_SPGEMM_COMMON_H