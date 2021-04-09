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
    IT flops = calculateFlops(A, B, M, flopsPerRow);

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
        auto[upperBoundSizeC, maxRowSizeA, maxRowSizeM] = scanInputs<true, true, true>(
                rowBeginIdx, rowEndIdx, flopsPerRow, A, B, M);

        // Initialize row algorithms
        MaskedHash<IT, NT> hash{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSizeHash, bufferAlignmentHash] = hash.getMemoryRequirement();

        MaskedSPA2A<IT, NT> spa{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSizeSPA, bufferAlignmentSPA] = spa.getMemoryRequirement();

        MaskedHeap_v1<IT, NT> heap{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSizeHeap, bufferAlignmentHeap] = heap.getMemoryRequirement();

        MaskIndexed<IT, NT> maskIndexed{B.cols, maxRowSizeA, maxRowSizeM};
        auto[bufferSizeMaskIndexed, bufferAlignmentMaskIndexed] = maskIndexed.getMemoryRequirement();

        auto bufferSize = std::max(bufferSizeHash, bufferSizeSPA);
        bufferSize = std::max(bufferSize, bufferSizeHeap);
        bufferSize = std::max(bufferSize, bufferSizeMaskIndexed);
//        bufferSize += 10000;

        auto bufferAlignment = std::lcm(bufferAlignmentHash, bufferAlignmentSPA);
        bufferAlignment = std::lcm(bufferAlignment, bufferSizeHeap);
        bufferAlignment = std::lcm(bufferAlignment, bufferAlignmentMaskIndexed);

        auto buffer = mallocAligned(bufferSize, bufferAlignment);
        size_t dirty = bufferSize;

        // Allocate temporary memory for C's column IDs and Values
        IT *colIdsLocal = my_malloc<IT>(upperBoundSizeC);
        NT *valuesLocal = my_malloc<NT>(upperBoundSizeC);
        IT *currColId = colIdsLocal;
        NT *currValue = valuesLocal;

        // Numeric phase
        uint8_t currAlg = 255;
        for (IT row = rowBeginIdx; row < rowEndIdx; ++row) {
            uint8_t newAlg = 1;
            if (newAlg != currAlg) {
                switch (currAlg) {
                    case 0:
                        dirty = hash.getNumericAccumulator().releaseBuffer();
                        break;
                    case 1:
                        dirty = spa.getNumericAccumulator().releaseBuffer();
                        break;
                    case 2:
                        dirty = heap.getNumericAccumulator().releaseBuffer();
                        break;
                    case 3:
                        dirty = maskIndexed.getNumericAccumulator().releaseBuffer();
                        break;
                }

                assert(dirty <= bufferSize);

                switch (newAlg) {
                    case 0:
                        hash.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
                        break;
                    case 1:
                        spa.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
                        break;
                    case 2:
                        heap.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
                        break;
                    case 3:
                        maskIndexed.getNumericAccumulator().setBuffer(buffer, bufferSize, dirty);
                        break;
                }
                currAlg = newAlg;
            }

            if (M.rowptr[row] != M.rowptr[row + 1]) {
                switch (currAlg) {
                    case 0:
                        hash.numericRow(A, B, M, multop, addop, row, currColId, currValue);
                        break;
                    case 1:
                        spa.numericRow(A, B, M, multop, addop, row, currColId, currValue);
                        break;
                    case 2:
                        heap.numericRow(A, B, M, multop, addop, row, currColId, currValue);
                        break;
                    case 3:
                        maskIndexed.numericRow(A, B, M, multop, addop, row, currColId, currValue);
                        break;
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

        my_free(colIdsLocal);
        my_free(valuesLocal);
        freeAligned(buffer);
    }

    my_free(flopsPerRow);
    my_free(cumulativeWork);
    my_free(rowNvals);
    my_free(threadsNvals);
}


#endif //MASKEDSPGEMM_MASKED_SPGEMM_POLY_H
