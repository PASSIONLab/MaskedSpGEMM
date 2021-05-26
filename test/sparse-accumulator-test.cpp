#include <iostream>

#include "../spgemm-blocks/util.h"
#include "../spgemm-blocks/msa/MaskedSparseAccumulator1A.h"

int main() {
    using KeyT = uint64_t;
    using AccumT = MaskedSparseAccumulator1A<KeyT, void>;
    const size_t maxIndex = 100;
    const size_t maxEntries = 80;

    auto spa = AccumT(maxIndex);

    auto [bufferSize, bufferAlignment] = spa.getMemoryRequirement();
    std::byte *buffer = mallocAligned(bufferSize, bufferAlignment);
    size_t dirtyMemory = bufferSize;
    spa.setBuffer(buffer, bufferSize, dirtyMemory);

//    for (int i = 0; i < maxEntries; i++) {
//        assert(spa.insert(i * maxIndex / maxEntries));
//    }
//
//    for (int i = 0; i < maxEntries; i++) {
//        assert(!spa.insert(i * maxIndex / maxEntries));
//    }

    for (int i = 0; i < maxEntries; i++) {
        spa.setAllowed(i * maxIndex / maxEntries);
    }

    for (int i = 0; i < maxEntries; i++) {
        assert(spa.erase(i * maxIndex / maxEntries));
    }

    for (int i = 0; i < maxEntries; i++) {
       assert(!spa.erase(i * maxIndex / maxEntries));
    }

    freeAligned(buffer);
}