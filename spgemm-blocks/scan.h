#ifndef MASKED_SPGEMM_SCAN_H
#define MASKED_SPGEMM_SCAN_H

#include <cstddef>
#include <omp.h>

#include "../utility.h"

template<class T>
void inclusiveScan(const T *src, size_t size, T *dst, int numThreads, T *partialSums) {
    int threadId = omp_get_thread_num();
    size_t elemPerThread = size / numThreads;
    size_t start = threadId * elemPerThread;
    size_t end = threadId + 1 != numThreads ? (threadId + 1) * elemPerThread : size;

    T localSum = 0;
    for (size_t i = start; i < end; ++i) {
        localSum += src[i];
    }
    partialSums[threadId] = localSum;

#pragma omp barrier
    T offset = 0;
    for (size_t i = 0; i < threadId; ++i) {
        offset += partialSums[i];
    }

    dst[start] = offset + src[start];
    for (size_t i = start + 1; i < end; ++i) {
        dst[i] = dst[i - 1] + src[i];
    }
}

template<class T>
void inclusiveScan(const T *src, size_t size, T *dst, int numThreads) {
    // If the input is small or if numThreads is 1, use sequential code
    if (size < (1u << 17u) || numThreads == 1) {
        dst[0] = src[0];
        for (size_t i = 1; i < size; ++i) {
            dst[i] = dst[i - 1] + src[i];
        }
        return;
    }

    T *partialSums = my_malloc<T>(numThreads, false);

#pragma omp parallel num_threads(numThreads) default(none) shared(src, size, dst, numThreads, partialSums)
    {
        inclusiveScan(src, size, dst, numThreads, partialSums);
    };

    my_free(partialSums);
}

template<class T>
void exclusiveScan(const T *src, size_t size, T *dst, int numThreads, T *partialSums) {
    int threadId = omp_get_thread_num();
    size_t elemPerThread = size / numThreads;
    size_t start = threadId * elemPerThread;
    size_t end = threadId + 1 != numThreads ? (threadId + 1) * elemPerThread : size;

    T localSum = 0;
    for (size_t i = start; i < end; ++i) {
        localSum += src[i];
    }
    partialSums[threadId] = localSum;

#pragma omp barrier
    T offset = 0;
    for (size_t i = 0; i < threadId; ++i) {
        offset += partialSums[i];
    }

    dst[start] = offset;
    for (size_t i = start + 1; i < end; ++i) {
        dst[i] = dst[i - 1] + src[i - 1];
    }
}

template<class T>
void exclusiveScan(const T *src, size_t size, T *dst, int numThreads) {
    // If the input is small or if numThreads is 1, use sequential code
    if (size < (1u << 1u) || numThreads == 1) {
        dst[0] = 0;
        for (size_t i = 1; i < size; ++i) {
            dst[i] = dst[i - 1] + src[i - 1];
        }
        return;
    }

    T *partialSums = my_malloc<T>(numThreads, false);

#pragma omp parallel num_threads(numThreads) default(none) shared(src, size, dst, numThreads, partialSums)
    {
        exclusiveScan(src, size, dst, numThreads, partialSums);
    };

    my_free(partialSums);
}

#endif //MASKED_SPGEMM_SCAN_H
