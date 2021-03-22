#ifndef MASKED_SPGEMM_SPARSE_ACCUMULATOR_2A_H
#define MASKED_SPGEMM_SPARSE_ACCUMULATOR_2A_H

#include <utility>
#include <limits>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "../util.h"


template<class S, class V>
struct SPA2AStorage {
    S *_states;
    V *_values;
};

template<class S>
struct SPA2AStorage<S, void> {
    S *_states;
};

template<class KeyT, class ValueT>
class SparseAccumulator2A {
protected:
    using T = std::make_unsigned_t<KeyT>;
    using StateT = uint8_t;

public:
    inline static const bool HAS_VALUE = !std::is_same_v<ValueT, void>;

    inline static const StateT EMPTY = std::numeric_limits<StateT>::max();
    inline static const StateT ALLOWED = 0;
    inline static const StateT INITIALIZED = 1;

//protected:
    const T _maxIndex;
    SPA2AStorage<StateT, ValueT> _storage;

    size_t _dirty;

public:
    SparseAccumulator2A(T maxIndex) : _maxIndex(maxIndex) {}

    SparseAccumulator2A(const SparseAccumulator2A &other) = delete;

    SparseAccumulator2A(SparseAccumulator2A &&other) = delete;

    SparseAccumulator2A &operator=(const SparseAccumulator2A &) = delete;

    SparseAccumulator2A &operator=(SparseAccumulator2A &&) = delete;

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        if constexpr (HAS_VALUE) {
            return {_maxIndex * (sizeof(StateT) + sizeof(ValueT)), std::lcm(sizeof(StateT), sizeof(ValueT))};
        } else {
            return {_maxIndex * sizeof(StateT), sizeof(StateT)};
        }
    }

    void setBuffer(std::byte *buffer, size_t bufferSize, const size_t dirty) {
        if constexpr (HAS_VALUE) {
            assert(isAligned(buffer, std::lcm(sizeof(StateT), sizeof(ValueT))));
            assert(_maxIndex * (sizeof(StateT) + sizeof(ValueT)) <= bufferSize);
        } else {
            assert(isAligned(buffer, sizeof(StateT)));
            assert(_maxIndex * sizeof(StateT) <= bufferSize);
        }

        _dirty = dirty;
        if constexpr (HAS_VALUE) {
            splitMemory(buffer, bufferSize, _dirty, _storage._values, _maxIndex, _storage._states, _maxIndex);
        } else {
            getCleanMemory(buffer, bufferSize, _dirty, _storage._states, _maxIndex);
        }
    }

    [[nodiscard]] size_t releaseBuffer() {
#if defined(DEBUG)
        _storage._states = nullptr;
        if constexpr (HAS_VALUE) { _storage._values = nullptr; }
#endif
        if constexpr (HAS_VALUE) {
            return std::max(_dirty, _maxIndex * sizeof(ValueT));
//        return _dirty;
        } else {
            return _dirty;
        }
    }

    [[nodiscard]] bool isEmpty(T idx) const {
        assert(0 <= idx && idx < _maxIndex);

        return _storage._states[idx] == EMPTY;
    }

    StateT &getState(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        return _storage._states[idx];
    }

    template<class U = ValueT>
    std::enable_if_t<!std::is_same_v<U, void>, ValueT> &getValue(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        return _storage._values[idx];
    }

    void setAllowed(KeyT key) {
        assert(0 <= key && key < _maxIndex);
        assert(_storage._states[key] == EMPTY);

        _storage._states[key] = ALLOWED;
    }

    bool erase(T idx) const {
        assert(0 <= idx && idx < _maxIndex);

        if (isEmpty(idx)) { return false; }
        _storage._states[idx] = EMPTY;
        return true;
    }

    void clear(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        if (_storage._states[idx] != EMPTY) {
            _storage._states[idx] = EMPTY;
//            if constexpr (HAS_VALUE) { memset(&_storage._values[idx], 0xFF, sizeof(ValueT)); }
        }
    }

    void clearAll() {
        memset(_storage._states, 0xFF, _maxIndex * sizeof(StateT));
    }

    bool isInitialized() const {
        StateT initialized;
        memset(&initialized, 0xFF, sizeof(StateT));

        for (size_t i = 0; i < _maxIndex; i++) {
            if (memcmp(&initialized, _storage._states + i, sizeof(StateT)) != 0) { return false; }
        }

        return true;
    }
};


#endif //MASKED_SPGEMM_SPARSE_ACCUMULATOR_H
