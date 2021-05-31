#ifndef MASKED_SPGEMM_TUPLE_UTIL_H
#define MASKED_SPGEMM_TUPLE_UTIL_H

#include <cstddef>
#include <random>
#include "../Triple.h"

template<class IT, class NT>
IT removeDuplicates(IT numValues, Triple<IT, NT> *triples) {
    // Sort and remove duplicates
    std::sort(triples, triples + numValues, [](auto lhs, auto rhs) {
        return lhs.row < rhs.row || (lhs.row == rhs.row && lhs.col < rhs.col);
    });

    IT dst = 1;
    for (IT src = 1; src < numValues; src++) {
        if (triples[src].row != triples[src - 1].row || triples[src].col != triples[src - 1].col) {
            triples[dst] = triples[src];
            dst++;
        }
    }
    return dst;
}

template<class IT, class NT>
IT generateRandomTriples(IT numRows, IT numCols, IT numValues, Triple<IT, NT> *&triples) {
    std::mt19937 gen;
    auto rowDist = std::uniform_int_distribution<IT>(0, numRows - 1);
    auto colDist = std::uniform_int_distribution<IT>(0, numCols - 1);

    triples = new Triple<IT, NT>[numValues];

    // Generate triples
    for (IT i = 0; i < numValues; i++) {
        IT row = rowDist(gen);
        IT col = colDist(gen);
        triples[i] = Triple(row, col, NT(1));
    }

    return removeDuplicates(numValues, triples);
}

#endif //MASKED_SPGEMM_TUPLE_UTIL_H
