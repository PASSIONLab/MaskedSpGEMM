#ifndef MASKED_SPGEMM_SPARSE_ACCUMULATOR_H
#define MASKED_SPGEMM_SPARSE_ACCUMULATOR_H

#include <utility>
#include <limits>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "../util.h"

template<class KeyT, class EntryT>
class SparseAccumulatorBase {
public:
    using StateT = decltype(std::declval<EntryT>().state);
    using T = std::make_unsigned_t<KeyT>;

    static_assert(std::is_unsigned_v<StateT> && !std::is_same_v<StateT, bool>);

public:
    inline static const StateT EMPTY = std::numeric_limits<StateT>::max();
    inline static const StateT ALLOWED = 0;
    inline static const StateT INITIALIZED = 1;

protected:
    const T _maxIndex;
    EntryT *_entries;

public:
    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        return {_maxIndex * sizeof(EntryT), sizeof(EntryT)};
    }

public:
    SparseAccumulatorBase(T maxIndex) : _maxIndex(maxIndex) {}

    SparseAccumulatorBase(const SparseAccumulatorBase &other) = delete;

    SparseAccumulatorBase(SparseAccumulatorBase &&other) = delete;

    SparseAccumulatorBase &operator=(const SparseAccumulatorBase &) = delete;

    SparseAccumulatorBase &operator=(SparseAccumulatorBase &&) = delete;

    void setBuffer(std::byte *buffer, size_t bufferSize, size_t dirty) {
        assert(isAligned(buffer, sizeof(EntryT)));
        assert(_maxIndex * sizeof(EntryT) <= bufferSize);
        // TODO: maybe use memorySplit
        _entries = reinterpret_cast<EntryT *>(buffer);
        memset(_entries, 0xFF, dirty);
    }

    void releaseBuffer(size_t &dirty) {
        _entries = nullptr;
    }

    [[nodiscard]] bool isEmpty(KeyT key) const {
        return _entries[key].state == EMPTY;
    }

    [[nodiscard]] EntryT &operator[](size_t key) {
        assert(0 <= key && key < _maxIndex);
        return _entries[key];
    }

    void setAllowed(KeyT key) {
        assert(0 <= key && key < this->_maxIndex);
        assert(this->_entries[key].state == EMPTY);

        this->_entries[key].state = ALLOWED;
    }

    bool erase(KeyT key) const {
        if (isEmpty(key)) { return false; }
        _entries[key].state = EMPTY;
        return true;
    }
};


template<class...>
struct SPAEntry;

// region SPA for numeric phase

template<class V1, class V2>
struct SPAEntry<V1, V2> {
    V1 state; /// 0 - allowed, data not initialized; 1 - data initialized; 255 - ignore (empty)
    V2 value;
};

template<class K, class V>
class SparseAccumulator : public SparseAccumulatorBase<K, SPAEntry<uint8_t, V>> {
    using super = SparseAccumulatorBase<K, SPAEntry<uint8_t, V>>;

public:
    SparseAccumulator(typename super::T maxIndex) : super(maxIndex) {}

    void clear(K key) {
        if (super::_entries[key].state != super::EMPTY) {
            super::_entries[key].state = super::EMPTY;
            memset(&super::_entries[key].value, 0xFF, sizeof(V));
        }
    }
};

// endregion

// region SPA for symbolic phase

template<class V>
struct SPAEntry<V> {
    V state; /// 0 - occupied; 255 - empty
};

template<class K>
class SparseAccumulator<K, void> : public SparseAccumulatorBase<K, SPAEntry<uint8_t>> {
    static_assert(sizeof(uint8_t) == 1);

    using super = SparseAccumulatorBase<K, SPAEntry<uint8_t>>;

public:
    SparseAccumulator(typename super::T maxIndex) : super(maxIndex) {}

    void clear(K key) {
        super::_entries[key].state = super::EMPTY;
    }
};

//endregion



#endif //MASKED_SPGEMM_SPARSE_ACCUMULATOR_H
