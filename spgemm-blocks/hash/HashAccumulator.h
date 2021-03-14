#ifndef MASKED_SPGEMM_HASHTABLE_H
#define MASKED_SPGEMM_HASHTABLE_H

#include <unistd.h>

#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>
#include <cassert>

#include "../../utility.h" // my_malloc

template<typename EntryT>
class HashAccumulatorBase {
public:
    using K = decltype(std::declval<EntryT>().key);
    using T = std::make_unsigned_t<K>;

public:
    inline static const T NOT_FOUND = std::numeric_limits<T>::max();
    inline static const K EMPTY = std::numeric_limits<K>::max();

    static const T SCALE = 107;
    static const T DEFAULT_CAPACITY = 16;

    static const std::ratio<2, 8> LOAD_FACTOR;

protected:
    T _capacity;
    T _size;
    T _mask;
    EntryT *_table;

    static T adjustSize(T requestedSize) {
        ++requestedSize;
        T size = DEFAULT_CAPACITY;
        while (size < requestedSize) { size <<= 1; }

        // (requestedSize / size) > LOAD_FACTOR (assumes LOAD_FACTOR < 2)
        while (requestedSize * LOAD_FACTOR.den > size * LOAD_FACTOR.num) { size <<= 1; }
        return size;
    }

public:
    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement(T capacity) {
        return {adjustSize(capacity) * sizeof(EntryT), sizeof(EntryT)};
    }

public:
    HashAccumulatorBase() : _table{nullptr} {};

    HashAccumulatorBase(T capacity, bool init = true) {
        _capacity = adjustSize(capacity);
        _size = _capacity;
        _mask = _size - 1;

        _table = my_malloc<EntryT>(_capacity);

        if (init) { reset(); }
    }

    HashAccumulatorBase(const HashAccumulatorBase &other) = delete;

    HashAccumulatorBase(HashAccumulatorBase &&other) = delete;

    HashAccumulatorBase &operator=(const HashAccumulatorBase &) = delete;

    HashAccumulatorBase &operator=(HashAccumulatorBase &&) = delete;

    void setBuffer(std::byte *buffer, size_t bufferSize, size_t dirty) {
        assert(isAligned(buffer, sizeof(EntryT)));
        _table = reinterpret_cast<EntryT*>(buffer);
        _capacity = bufferSize / sizeof(EntryT);

        /* check this */
        init();
    }

    void releaseBuffer(size_t &dirty) {
        _table = nullptr;
    }

    void resize(T size) {
        _size = adjustSize(size);
        _mask = _size - 1;
        assert(_size <= _capacity);
    }

    void init() {
        for (size_t i = 0; i < _capacity; ++i) { _table[i].key = EMPTY; }
    }

    void reset() {
        for (size_t i = 0; i < _size; ++i) { _table[i].key = EMPTY; }
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

    HashAccumulator() = default;

    [[deprecated]]
    HashAccumulator(typename super::T capacity, bool init = true) : super(capacity, init) {}

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

    HashAccumulator() = default;

    [[deprecated]]
    HashAccumulator(typename super::T capacity, bool init = true) : super(capacity, init) {}

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

    HashAccumulator() = default;

    [[deprecated]]
    HashAccumulator(typename super::T capacity, bool init = true) : super(capacity, init) {}

    bool insert(K key) {
        auto idx = super::findIdx(key);
        if (this->_table[idx].key != this->EMPTY) { return false; }

        this->_table[idx].key = key;
        return true;
    }
};

//endregion

#endif //MASKED_SPGEMM_HASHTABLE_H
