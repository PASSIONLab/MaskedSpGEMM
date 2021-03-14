#include <iostream>
#include <random>
#include "../spgemm-blocks/util.h"

int main() {
    const size_t niter = 100;
    const size_t maxSize = 1024 * 1024;
    const size_t maxAlignment = 128;

    std::mt19937 gen;
    auto sizeDist = std::uniform_int_distribution<size_t>(1, maxSize);
    auto alignmentDist = std::uniform_int_distribution<size_t>(1, maxAlignment);

    for (size_t i = 0; i < niter; i++) {
        size_t size = sizeDist(gen);
        size_t alignment = alignmentDist(gen);
        auto ptr = mallocAligned(size, alignment);

        for (int j = 0; j < size; j++) { reinterpret_cast<uint8_t *>(ptr)[j] = 1; }

        assert(isAligned(ptr, alignment));
        freeAligned(ptr);
    }
}