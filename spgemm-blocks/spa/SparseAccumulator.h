#ifndef MASKED_SPGEMM_SPARSE_ACCUMULATOR_H
#define MASKED_SPGEMM_SPARSE_ACCUMULATOR_H

#include <utility>
#include <limits>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "../util.h"

template<class S, class V>
struct SPAEntry {
    S state;
    V value;
};

template<class S>
struct SPAEntry<S, void> {
    S state;
};

template<class KeyT, class ValueT>
class SparseAccumulator {
protected:
    using T = std::make_unsigned_t<KeyT>;
    using StateT = uint8_t;

    using EntryT = SPAEntry<StateT, ValueT>;

public:
    inline static const StateT EMPTY = std::numeric_limits<StateT>::max();
    inline static const StateT ALLOWED = 0;
    inline static const StateT INITIALIZED = 1;

protected:
    const T _maxIndex;
    EntryT *_entries;

    size_t _dirty;

public:
    SparseAccumulator(T maxIndex) : _maxIndex(maxIndex) {}

    SparseAccumulator(const SparseAccumulator &other) = delete;

    SparseAccumulator(SparseAccumulator &&other) = delete;

    SparseAccumulator &operator=(const SparseAccumulator &) = delete;

    SparseAccumulator &operator=(SparseAccumulator &&) = delete;

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        return {_maxIndex * sizeof(EntryT), sizeof(EntryT)};
    }

    void setBuffer(std::byte *buffer, size_t bufferSize, const size_t dirty) {
        assert(isAligned(buffer, sizeof(EntryT)));
        assert(_maxIndex * sizeof(EntryT) <= bufferSize);

        _dirty = dirty;
        getCleanMemory(buffer, bufferSize, _dirty, _entries, _maxIndex);
    }

    [[nodiscard]] size_t releaseBuffer() {
#if defined(DEBUG)
        _entries = nullptr;
#endif
        return _dirty;
    }

    [[nodiscard]] bool isEmpty(T idx) const {
        assert(0 <= idx && idx < _maxIndex);

        return _entries[idx].state == EMPTY;
    }

    [[nodiscard]] EntryT &operator[](T idx) {
        assert(0 <= idx && idx < _maxIndex);

        return _entries[idx];
    }

    void setAllowed(KeyT key) {
        assert(0 <= key && key < this->_maxIndex);
        assert(this->_entries[key].state == EMPTY);

        this->_entries[key].state = ALLOWED;
    }

    bool erase(T idx) const {
        assert(0 <= idx && idx < _maxIndex);

        if (isEmpty(idx)) { return false; }
        _entries[idx].state = EMPTY;
        return true;
    }

    void clear(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        if (_entries[idx].state != EMPTY) {
            _entries[idx].state = EMPTY;
            if constexpr (!std::is_same_v<ValueT, void>) {
                memset(&_entries[idx].value, 0xFF, sizeof(ValueT));
            }
        }
    }

    void clearAll() {
        memset(_entries, 0xFF, _maxIndex * sizeof(EntryT));
    }

    bool isInitialized() const {
        EntryT initialized;
        memset(&initialized, 0xFF, sizeof(EntryT));

        for (size_t i = 0; i < _maxIndex; i++) {
            if (memcmp(&initialized, _entries + i, sizeof(EntryT)) != 0) { return false; }
        }

        return true;
    }
};

#endif //MASKED_SPGEMM_SPARSE_ACCUMULATOR_H
