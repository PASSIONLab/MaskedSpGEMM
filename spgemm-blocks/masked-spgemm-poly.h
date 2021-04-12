#ifndef MASKEDSPGEMM_MASKED_SPGEMM_POLY_H
#define MASKEDSPGEMM_MASKED_SPGEMM_POLY_H

template<class IT, class NT, class MultiplyOperation, class AddOperation>
void MaskedSpGEMM1p(const CSR<IT, NT> &A, const CSR<IT, NT> &B, CSR<IT, NT> &C, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, unsigned numThreads = 0) {

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
        auto[upperBoundSizeC, maxRowSizeA, maxRowSizeM] = scanInputs(rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithms
//        MaskedHash<IT, NT> hash{B.cols, maxRowSizeA, maxRowSizeM};
//        auto[bufferSizeHash, bufferAlignmentHash] = hash.getMemoryRequirement();

        MaskedSPA2A<IT, NT> spa{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSizeSPA, bufferAlignmentSPA] = spa.getMemoryRequirement();

//        MaskedHeap_v1<IT, NT> heap{B.cols, maxRowSizeA, maxRowSizeM};
//        auto[bufferSizeHeap, bufferAlignmentHeap] = heap.getMemoryRequirement();

        MaskIndexed<IT, NT> maskIndexed{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSizeMaskIndexed, bufferAlignmentMaskIndexed] = maskIndexed.getMemoryRequirement();

        size_t bufferSize = 0;
//        bufferSize = std::max(bufferSize, bufferSizeHash);
        bufferSize = std::max(bufferSize, bufferSizeSPA);
//        bufferSize = std::max(bufferSize, bufferSizeHeap);
        bufferSize = std::max(bufferSize, bufferSizeMaskIndexed);

        size_t bufferAlignment = 1;
//        bufferAlignment = std::lcm(bufferAlignment, bufferAlignmentHash);
        bufferAlignment = std::lcm(bufferAlignment, bufferAlignmentSPA);
//        bufferAlignment = std::lcm(bufferAlignment, bufferSizeHeap);
        bufferAlignment = std::lcm(bufferAlignment, bufferAlignmentMaskIndexed);

        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        spa.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
        maskIndexed.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);

        // Allocate temporary memory for C's column IDs and Values
        IT *colIdsLocal = my_malloc<IT>(upperBoundSizeC);
        NT *valuesLocal = my_malloc<NT>(upperBoundSizeC);
        IT *currColId = colIdsLocal;
        NT *currValue = valuesLocal;

        // Numeric phase
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            if (flopsPerRow[row] != 0) {
                if ((M.rowptr[row + 1] - M.rowptr[row]) * (A.rowptr[row + 1] - A.rowptr[row]) < flopsPerRow[row]) {
                    maskIndexed.numericRow(A, B, M, multop, addop, row, currColId, currValue);
                } else {
                    spa.numericRow(A, B, M, multop, addop, row, currColId, currValue);
                }
            } else {
                rowNvals[row] = 0;
            }
        }
        threadsNvals[thisThread] = currColId - colIdsLocal;
//        dirty = alg.getNumericAccumulator().releaseBuffer();

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


#endif //MASKEDSPGEMM_MASKED_SPGEMM_POLY_H
