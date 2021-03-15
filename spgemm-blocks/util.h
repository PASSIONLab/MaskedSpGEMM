
#ifndef MASKED_SPGEMM_UTIL_H
#define MASKED_SPGEMM_UTIL_H

#include <cstring>
#include <algorithm>
#include <cassert>

template<class T>
bool isAligned(const T *ptr, size_t alignment) noexcept {
    auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
    return iptr % alignment == 0;
}

template<class T>
bool isAligned(const T *ptr) noexcept {
    return isAligned(ptr, sizeof(T));
}

template<class T = std::byte>
T *mallocAligned(size_t nelem, size_t alignment = sizeof(T)) {
    size_t nbytes = nelem * sizeof(T);
    auto mem = static_cast<std::byte *>(std::malloc(alignment + sizeof(size_t) + nbytes));

    size_t shiftSize = sizeof(size_t);
    size_t rem = reinterpret_cast<intptr_t>(mem + shiftSize) % alignment;
    shiftSize += (rem != 0) ? (alignment - rem) : 0;
    mem += shiftSize;

    *reinterpret_cast<size_t *>(mem - sizeof(size_t)) = shiftSize;

    return reinterpret_cast<T *>(mem);
};

template<class T>
void freeAligned(T *ptr) {
    auto *mem = reinterpret_cast<std::byte *>(ptr);
    size_t shiftSize = *reinterpret_cast<size_t *>(mem - sizeof(size_t));
    mem -= shiftSize;
    std::free(mem);
}

template<size_t div, class T>
T roundUp(T num) {
    T remainder = num % div;
    return remainder != 0 ? num + (div - remainder) : num;
}

template<size_t div, class T>
T roundDown(T num) {
    return num - num % div;
}

template<class DirtyT, class CleanT>
void splitMemory(std::byte *mem, size_t memSize, size_t &dirtyMemSize,
                   DirtyT *&dirtyObjsMem, size_t dirtyObjsNum, CleanT *&cleanObjsMem, size_t cleanObjsNum) {
    const size_t dirtyObjsMemSize = sizeof(DirtyT) * dirtyObjsNum;
    const size_t cleanObjsMemSize = sizeof(CleanT) * cleanObjsNum;
    memSize = roundDown<sizeof(CleanT)>(memSize);
    size_t usedMem = roundUp<sizeof(CleanT)>(std::max(dirtyMemSize, dirtyObjsMemSize));

    // Check if we need more memory than we have
    if (cleanObjsMemSize > memSize - usedMem) {
        assert(dirtyMemSize > dirtyObjsMemSize);
        size_t toClean = cleanObjsMemSize - (memSize - dirtyMemSize);

        // Align the clean memory
        toClean += (dirtyMemSize - toClean) % sizeof(CleanT);

        assert(dirtyMemSize >= toClean);
        memset(mem + dirtyMemSize - toClean, 0xFF, toClean);
        usedMem = dirtyMemSize - toClean;

        dirtyMemSize -= toClean;
    }

    dirtyObjsMem = reinterpret_cast<DirtyT *>(mem);
    cleanObjsMem = reinterpret_cast<CleanT *>(mem + usedMem);

    assert(static_cast<void *>(dirtyObjsMem + dirtyObjsNum) <= static_cast<void *>(cleanObjsMem));
    assert(static_cast<void *>(cleanObjsMem + cleanObjsNum) <= static_cast<void *>(mem + memSize));
}

template<class CleanT>
void getCleanMemory(std::byte *mem, size_t memSize, size_t &dirtySize, CleanT *&cleanObjsMem, size_t cleanObjsNum) {
    std::byte *dummy;
    splitMemory(mem,memSize, dirtySize, dummy, 0, cleanObjsMem, cleanObjsNum);
}


#endif //MASKED_SPGEMM_UTIL_H
