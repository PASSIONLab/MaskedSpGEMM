#ifndef MASKED_SPGEMM_COMMON_H
#define MASKED_SPGEMM_COMMON_H

void setNumThreads(unsigned &numThreads) {
    if (numThreads == 0) {
#pragma omp parallel default(none) shared(numThreads)
#pragma omp single
        numThreads = omp_get_num_threads();
    }
}

template<class IT, class NT,
        template<class, class> class AT,
        template<class, class> class BT,
        template<class, class> class CT,
        template<class, class> class MT>
void verifyInputs(const AT<IT, NT> &A, const BT<IT, NT> &B, const CT<IT, NT> &C, const MT<IT, NT> &M) {
    assert(A.cols == B.rows);
    assert(M.rows == A.rows);
    assert(M.cols == B.cols);
}

template<typename IT, typename NT>
IT calculateMultOps(const CSR<IT, NT> &A, const CSR<IT, NT> &B, int numThreads = 0) {
    IT nops = 0; // total multiplications needed to generate C

    if (numThreads == 0) {
        numThreads = omp_get_max_threads();
    }

    #pragma omp parallel for reduction(+:nops) num_threads(numThreads)
    for (IT i = 0; i < A.rows; ++i) {
        IT nopsRow = 0;
        for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
            IT inner = A.colids[j];
            IT npins = B.rowptr[inner + 1] - B.rowptr[inner];
            nopsRow += npins;
        }
        nops += nopsRow;
    }

    return nops;
}

template<typename IT, typename NT>
IT calculateFlops(const CSR<IT, NT> &A, const CSR<IT, NT> &B, IT *flopsPerRow, int numThreads) {
    IT flops = 0; // total flop (multiplication) needed to generate C

#pragma omp parallel for reduction(+:flops) num_threads(numThreads)
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

template<typename IT, typename NT>
IT calculateFlops(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M, IT *flopsPerRow, int numThreads) {
    IT flops = 0; // total flop (multiplication) needed to generate C

#pragma omp parallel for reduction(+:flops) num_threads(numThreads)
    for (IT i = 0; i < A.rows; ++i) {
        IT flopsRow = 0;
        if (M.rowptr[i + 1] != M.rowptr[i]) {
            for (IT j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
                IT inner = A.colids[j];
                IT npins = B.rowptr[inner + 1] - B.rowptr[inner];
                flopsRow += npins;
            }
        }
        flopsPerRow[i] = flopsRow;
        flops += flopsRow;
    }

    return flops;
}

template<typename IT, typename NT>
IT calculateWork(const CSR<IT, NT> &A, const CSC<IT, NT> &B, const CSR<IT, NT> &M, IT *workPerRow, int numThreads) {
    IT work = 0;

#pragma omp parallel for reduction(+:work) num_threads(numThreads)
    for (IT i = 0; i < M.rows; ++i) {
        IT workRow = 0;
        for (IT j = M.rowptr[i]; j < M.rowptr[i + 1]; ++j) {
            IT lenA = A.rowptr[i + 1] - A.rowptr[i];
            IT lenB = B.colptr[M.colids[j] + 1] - B.colptr[M.colids[j]];
            if (lenA != 0 && lenB != 0) { workRow += lenA + lenB; }
        }
        workPerRow[i] = workRow;
        work += workRow;
    }

    return work;
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

template<bool complemented, bool calcUpperBoundSizeC, bool calcMaxRowSizeA, bool calcMaxRowSizeM, bool calcMaxRowFlops, class IT, class NT>
std::tuple<IT, IT, IT, IT> scanInputs(IT rowBeginIdx, IT rowEndIdx, IT *flopsPerRow,
                                  const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M) {
    IT sizeC = 0;
    IT maxRowSizeM = 0;
    IT maxRowSizeA = 0;
    IT maxRowFlops = 0;

    for (IT row = rowBeginIdx; row < rowEndIdx; row++) {
        if (calcUpperBoundSizeC) {
            if (!complemented) { sizeC += std::min(flopsPerRow[row], M.rowptr[row + 1] - M.rowptr[row]); }
            else { sizeC += flopsPerRow[row]; }
        }
        if (calcMaxRowSizeA) { maxRowSizeA = std::max(maxRowSizeA, A.rowptr[row + 1] - A.rowptr[row]); }
        if (calcMaxRowSizeM) { maxRowSizeM = std::max(maxRowSizeM, M.rowptr[row + 1] - M.rowptr[row]); }
        if (calcMaxRowFlops) { maxRowFlops = std::max(maxRowFlops, flopsPerRow[row]); }
    }

    return {sizeC, maxRowSizeA, maxRowSizeM, maxRowFlops};
}

template<class IT, class NT,
        template<class, class> class AT,
        template<class, class> class BT,
        template<class, class> class CT>
void initC(const AT<IT, NT> &A, const BT<IT, NT> &B, CT<IT, NT> &C, IT *threadsNvals, int numThreads) {
    // If C == A || C == B || C == M
    auto nrows = A.rows;
    auto ncols = B.cols;

    C.make_empty();

    C.rows = nrows;
    C.cols = ncols;
    C.rowptr = my_malloc<IT>(C.rows + 1, false);

    C.nnz = std::accumulate(threadsNvals, threadsNvals + numThreads, IT(0));
    C.colids = my_malloc<IT>(C.nnz, false);
    C.values = my_malloc<NT>(C.nnz, false);
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
