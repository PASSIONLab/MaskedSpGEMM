#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <numeric>
#include <memory>
#include "../spgemm-blocks/util.h"

template<class DirtyT, class CleanT>
void test(size_t memSize) {
    const size_t alignment = std::lcm(sizeof(DirtyT), sizeof(CleanT));
    auto mem = mallocAligned<std::byte>(memSize, alignment);

    for (size_t i = 0; i * sizeof(DirtyT) < memSize; i++) {
        for (size_t j = 0;
             roundUp<sizeof(CleanT)>(i * sizeof(DirtyT)) + roundUp<sizeof(DirtyT)>(j * sizeof(CleanT)) < memSize; j++) {
            for (size_t k = 0; roundUp<sizeof(CleanT)>(k) < memSize; k++) {
                const size_t dirtyNbytes = k;
                const size_t cleanNbytes = memSize - k;

                memset(mem, 0x55, dirtyNbytes);
                memset(mem + dirtyNbytes, 0xFF, cleanNbytes);


                DirtyT *dirtyMem;
                CleanT *cleanMem;
                size_t cleaned = splitMemory(mem, memSize, dirtyNbytes, dirtyMem, i, cleanMem, j);

                // Check if returned clean memory is clean
                for (size_t l = 0; l < j; l++) {
                    assert(reinterpret_cast<uint8_t *>(cleanMem)[l] == 0xFF);
                }

                // Check if memory that's not declared dirty is clean
                for (size_t l = dirtyNbytes; l < memSize; l++) {
                    assert(reinterpret_cast<uint8_t *>(mem)[l] == 0xFF);
                }

                // check if splitMemory cleaned correct bytes
                for (size_t l = dirtyNbytes - cleaned; l < dirtyNbytes; l++) {
                    assert(reinterpret_cast<uint8_t *>(mem)[l] == 0xFF);
                }

                // check if splitMemory cleaned correct bytes
                for (size_t l = 0; l < dirtyNbytes - cleaned; l++) {
                    assert(reinterpret_cast<uint8_t *>(mem)[l] == 0x55);
                }

                // if there was enough memory, splitMemory should have not cleaned anything
                assert(cleanNbytes >= j * sizeof(CleanT) || cleaned != 0);

                // Check if he the output is aligned
                assert(isAligned(dirtyMem) || i == 0);
                assert(isAligned(cleanMem) || j == 0);
            }
        }
    }

    freeAligned(mem);
}

struct S1 {
    uint8_t a[3];
};

int main() {
    test<uint8_t, uint8_t>(15);
    test<uint8_t, uint16_t>(15);
    test<uint16_t, uint8_t>(15);
    test<uint16_t, uint16_t>(15);

    test<uint8_t, uint32_t>(15);
    test<uint32_t, uint8_t>(15);

    test<S1, S1>(15);
    test<uint8_t, S1>(15);
    test<S1, uint8_t>(15);

    test<uint16_t, S1>(15);
    test<S1, uint16_t>(15);

    test<uint32_t, S1>(15);
    test<S1, uint32_t>(15);
}

