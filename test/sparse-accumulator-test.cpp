#include <iostream>

#include "../spgemm-blocks/util.h"
#include "../spgemm-blocks/spa/SparseAccumulator.h"

int main() {
    using KeyT = uint64_t;
    using AccumT = SparseAccumulator<KeyT>;
    const size_t maxIndex = 100;
    const size_t maxEntries = 80;


    size_t bufferSize = AccumT::requiredMemory(maxIndex, maxEntries);
    size_t bufferAlignment = AccumT::requiredAlignment();
    std::byte *buffer = mallocAligned(bufferSize, bufferAlignment);

    size_t dirtyMemory = bufferSize;
    auto spa = SparseAccumulator<KeyT>(maxIndex, maxEntries, buffer, bufferSize, dirtyMemory);

    for (int i = 0; i < maxEntries; i++) {
        assert(spa.insert(i * maxIndex / maxEntries));
    }

    for (int i = 0; i < maxEntries; i++) {
        assert(!spa.insert(i * maxIndex / maxEntries));
    }

    for (int i = 0; i < maxEntries; i++) {
        assert(spa.erase(i * maxIndex / maxEntries));
    }

    for (int i = 0; i < maxEntries; i++) {
       assert(!spa.erase(i * maxIndex / maxEntries));
    }

    freeAligned(buffer);

}