#include <random>
#include <set>
#include <map>

#include "../spgemm-blocks/hash/HashAccumulator.h"


template<template<class, class> class HashTableT, class K, class V>
void testAllOps(size_t nops, size_t niter, K maxKey) {
    using ValueT = std::conditional_t<std::is_void_v<V>, int, V>;

    std::mt19937 gen;
    auto opDist = uniform_int_distribution<K>(0, 2);
    auto keyDist = uniform_int_distribution<K>(0, maxKey);
    auto valueDist = uniform_int_distribution<ValueT>(0, std::numeric_limits<ValueT>::max());


    for (size_t iter = 0; iter < niter; ++iter) {
        std::conditional_t<std::is_void_v<V>, std::map<K, bool>, std::map<K, V>> map;
        HashTableT<K, V> hashTable{std::make_unsigned_t<K>(maxKey * 3 / 2)};

        for (size_t i = 0; i < nops; i++) {
            auto op = opDist(gen);
            auto key = keyDist(gen); // (i * 5) * 16
            auto value = valueDist(gen);

            switch (op) {
                case 0: {
                    // insert op
                    size_t cnt = map.count(key);
                    if (cnt == 0) { map[key] = value; }

                    bool inserted;
                    if constexpr (std::is_void_v<V>) {
                        inserted = hashTable.insert(key);
                    } else {
                        inserted = hashTable.insert(key, value);
                    }
                    if (inserted != (cnt == 0)) {
                        std::cerr << "Insertion error. Key: \"" << key << "\"" << std::endl;
                    }

                    break;
                }
                case 1: {
                    // erase op
                    size_t cnt = map.erase(key);
                    bool deleted = hashTable.erase(key);
                    if (deleted != (cnt != 0)) {
                        std::cerr << "Deleting error. Key: \"" << key << "\"" << std::endl;
                    }

                    break;
                }
                case 2: {
                    // find op
                    size_t cnt = map.count(key);
                    auto idx = hashTable.find(key);
                    bool found = idx != HashTableT<K, V>::NOT_FOUND;

                    if (found != (cnt != 0)) { std::cerr << "Search error. Key: \"" << key << "\"" << std::endl; }

                    if constexpr (!std::is_void_v<V>) {
                        if (found && map[key] != hashTable.operator[](idx).value1) {
                            std::cerr << "Search error. Key: \"" << key << "\"" << std::endl;
                        }
                    }
                    break;
                }
                default: {
                    std::cerr << "Internal error." << std::endl;
                    break;
                }
            }
        }
    }
}

int main() {
    testAllOps<HashAccumulator, int, void>(1000000, 10, 5);
    testAllOps<HashAccumulator, int, int>(1000000, 10, 100);

//    using T = unsigned ;
//
//    MaskedHashTable<T, void, void> ht(10);
//int off = 1;
//    ht.insert(0 + off);
//    ht.insert(16 + off);
//    ht.insert(32 + off);
//
//    std::cout << ht.find(0 + off) << std::endl;
//    std::cout << ht.find(16 + off) << std::endl;
//    std::cout << ht.find(32 + off) << std::endl;
//
//    ht.erase(16 + off);
//
//    std::cout << ht.find(0 + off) << std::endl;
//    std::cout << ht.find(16 + off) << std::endl;
//    std::cout << ht.find(32 + off) << std::endl;


}