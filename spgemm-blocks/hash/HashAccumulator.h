#ifndef MASKED_SPGEMM_HASHTABLE_H
#define MASKED_SPGEMM_HASHTABLE_H

#include <unistd.h>

#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>
#include <cassert>
#include <tuple>

#include "../../utility.h" // my_malloc
#include "../util.h"

template<typename EntryT>
class HashAccumulatorBase {
public:
    using K = decltype(std::declval<EntryT>().key);
    using T = std::make_unsigned_t<K>;

public:
    inline static const T NOT_FOUND = std::numeric_limits<T>::max();
    inline static const K EMPTY = std::numeric_limits<T>::max();

    static const T SCALE = 107;
    static const T DEFAULT_CAPACITY = 16;

    using LOAD_FACTOR = std::ratio<2, 8>;

protected:
    T _capacity;
    T _size;
    T _mask;
    EntryT *_table;

    size_t _dirty;
    size_t _maxSize;

    static T adjustSize(T requestedSize) {
        ++requestedSize;
        T size = DEFAULT_CAPACITY;
        while (size < requestedSize) { size <<= 1; }

        // (requestedSize / size) > LOAD_FACTOR (assumes LOAD_FACTOR < 2)
        while (requestedSize * LOAD_FACTOR::den > size * LOAD_FACTOR::num) { size <<= 1; }
        return size;
    }

public:
    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        return {_capacity * sizeof(EntryT), sizeof(EntryT)};
    }

public:
    explicit HashAccumulatorBase(size_t maxRowSizeM)
            : _capacity(adjustSize(maxRowSizeM)), _size(0), _mask(0), _table{nullptr}, _dirty(0), _maxSize(0) {};

    HashAccumulatorBase(const HashAccumulatorBase &other) = delete;

    HashAccumulatorBase(HashAccumulatorBase &&other) = delete;

    HashAccumulatorBase &operator=(const HashAccumulatorBase &) = delete;

    HashAccumulatorBase &operator=(HashAccumulatorBase &&) = delete;

    void setBuffer(std::byte *buffer, size_t bufferSize, const size_t dirty) {
        assert(isAligned(buffer, sizeof(EntryT)));
        assert(_capacity * sizeof(EntryT) <= bufferSize);

        _dirty = dirty;
        getCleanMemory(buffer, bufferSize, _dirty, _table, _capacity);
    }

    [[nodiscard]] size_t releaseBuffer() {
#if defined(DEBUG)
        _table = nullptr;
//        _capacity = 0;
        _size = 0;
        _mask = 0;
#endif
        return _dirty + _maxSize * sizeof(EntryT);
    }

    void resize(T size) {
        _size = adjustSize(size);
        _mask = _size - 1;
        assert(_size <= _capacity);

        if (_size > _maxSize) { _maxSize = _size; }
    }

    void init() {
        for (size_t i = 0; i < _capacity; ++i) { _table[i].key = EMPTY; }
    }

    void reset() {
        for (size_t i = 0; i < _size; ++i) { _table[i].key = EMPTY; }
    }

    void clean() {
        memset(_table, 0xFF, _maxSize * sizeof(EntryT));
    }

    T find(K key) const {
        T idx = findIdx(key);
        return _table[idx].key == EMPTY ? NOT_FOUND : idx;
    }

    EntryT &operator[](size_t idx) {
        return _table[idx];
    }

    bool erase(K key) {
        T eraseIdx = findIdx(key);

        if (_table[eraseIdx].key == EMPTY) { return false; }

        T shiftIdx = eraseIdx;
        while (true) {
            shiftIdx = (shiftIdx + 1) & _mask;
            if (_table[shiftIdx].key == EMPTY) { break; }

            T insertIdx = hash(_table[shiftIdx].key) & _mask;

            if ((insertIdx <= eraseIdx && eraseIdx <= shiftIdx) ||
                (eraseIdx <= shiftIdx) && (shiftIdx < insertIdx) ||
                (shiftIdx < insertIdx) && (insertIdx <= eraseIdx)) {
                _table[eraseIdx] = _table[shiftIdx];
                eraseIdx = shiftIdx;
            }
        }

        _table[eraseIdx].key = EMPTY;
        return true;
    }

protected:
    T findIdx(K key) const {
        T hv = (key * SCALE) & _mask;
        while (_table[hv].key != EMPTY && _table[hv].key != key) { hv = (hv + 1) & _mask; }
        return hv;
    }

    // return idx of the key that's guaranteed to be in the hashmap
    T findIdxForce(K key) const {
        T hv = (key * SCALE) & _mask;
        while (_table[hv].key != key) { hv = (hv + 1) & _mask; }
        return hv;
    }

    [[gnu::always_inline]]
    T hash(K key) const {
        return (key * SCALE) & _mask;
    }
};

template<class K, class V1 = void, class V2 = void>
struct HashAccumulator;

//region KVV

template<class K, class V1 = void, class V2 = void>
struct HashAccumEntryT {
    K key;
    V1 value1;
    V2 value2;
};

template<class K, class V1>
struct HashAccumulator<K, V1, bool> : public HashAccumulatorBase<HashAccumEntryT<K, V1, bool>> {

    using super = HashAccumulatorBase<HashAccumEntryT<K, V1, bool>>;

    HashAccumulator(size_t maxRowSizeM) : super(maxRowSizeM) {};

    bool insert(K key, V1 value1, bool value2) {
        auto idx = super::findIdx(key);
        if (this->_table[idx].key != this->EMPTY) { return false; }

        this->_table[idx].key = key;
        this->_table[idx].value1 = value1;
        this->_table[idx].value2 = value2;
        return true;
    }

    bool insert(K key) {
        auto idx = super::findIdx(key);
        if (this->_table[idx].key != this->EMPTY) { return false; }

        this->_table[idx].key = key;
        this->_table[idx].value2 = false;
        return true;
    }

    // gather valid values
    template<typename IT, typename NT>
    void gather(IT *&idxPtr, NT *&valPtr) {
        for (IT i = 0; i < this->_size; ++i) {
            auto &elem = this->_table[i];
            if (elem.key != this->EMPTY && elem.value2) {
                *idxPtr = elem.key;
                *valPtr = elem.value1;
                ++idxPtr;
                ++valPtr;
                memset(&elem.value1, 0xFF, sizeof(V1));
            }
            elem.key = this->EMPTY;
        }
    }

    template<typename IT, typename NT>
    void gather(IT *&idxPtr, NT *&valPtr, const IT *keysBegin, const IT *keysEnd) {
        for (auto keysIt = keysBegin; keysIt != keysEnd; ++keysIt) {
            auto idx = this->findIdxForce(*keysIt);
            auto &elem = this->_table[idx];

            if (elem.value2) {
                *idxPtr = elem.key;
                *valPtr = elem.value1;
                ++idxPtr;
                ++valPtr;
                memset(&elem.value1, 0xFF, sizeof(V1));
            }
            elem.key = this->EMPTY;
        }
    }
};

//endregion

//region KV

template<class K, class V1>
struct HashAccumEntryT<K, V1, void> {
    K key;
    V1 value1;
};

template<class K, class V1>
struct HashAccumulator<K, V1, void> : public HashAccumulatorBase<HashAccumEntryT<K, V1>> {

    using super = HashAccumulatorBase<HashAccumEntryT<K, V1>>;

    HashAccumulator(size_t maxRowSizeM) : super(maxRowSizeM) {};

    bool insert(K key, V1 value1) {
        auto idx = super::findIdx(key);
        if (this->_table[idx].key != this->EMPTY) { return false; }

        this->_table[idx].key = key;
        this->_table[idx].value1 = value1;
        return true;
    }

    // gather valid values
    template<typename IT, typename NT>
    void gather(IT *idx_ptr, NT *val_ptr) {
        for (auto &it : this->_table) {
            if (it.key != this->EMPTY) {
                *idx_ptr = it.key;
                *val_ptr = it.value;
                ++idx_ptr;
                ++val_ptr;
            }
            it = this->EMPTY;
        }
    }
};

//endregion

//region K

template<class K>
struct HashAccumEntryT<K, void, void> {
    K key;
};

template<class K>
struct HashAccumulator<K, void, void> : public HashAccumulatorBase<HashAccumEntryT<K>> {

    using super = HashAccumulatorBase<HashAccumEntryT<K>>;

    HashAccumulator(size_t maxRowSizeM) : super(maxRowSizeM) {};

    bool insert(K key) {
        auto idx = super::findIdx(key);
        if (this->_table[idx].key != this->EMPTY) { return false; }

        this->_table[idx].key = key;
        return true;
    }
};

//endregion

#endif //MASKED_SPGEMM_HASHTABLE_H
