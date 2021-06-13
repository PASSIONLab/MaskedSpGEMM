#ifndef MASKED_SPGEMM_SPARSE_ACCUMULATOR_2A_H
#define MASKED_SPGEMM_SPARSE_ACCUMULATOR_2A_H

#include <utility>
#include <limits>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>

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

template<class KeyT, class ValueT, bool Complemented>
class MaskedSparseAccumulator2A {
protected:
    using T = std::make_unsigned_t<KeyT>;
    using StateT = uint8_t;

public:
    inline static const bool HAS_VALUE = !std::is_same_v<ValueT, void>;

    // Initial value (default value) for the states is 0b11....11.
    inline static const StateT NOT_ALLOWED = !Complemented ? std::numeric_limits<StateT>::max() : 0;
    inline static const StateT ALLOWED = !Complemented ? 0 : std::numeric_limits<StateT>::max();
    inline static const StateT INITIALIZED = 1;

    inline static const StateT DEFAULT_STATE = !Complemented ? NOT_ALLOWED : ALLOWED;

protected:
    const T _maxIndex;
    SPA2AStorage<StateT, ValueT> _storage;

    size_t _dirty;
    std::vector<KeyT> _dirtyIndices; // TODO: replace with an array

public:
    MaskedSparseAccumulator2A(T maxIndex) : _maxIndex(maxIndex) {}

    MaskedSparseAccumulator2A(const MaskedSparseAccumulator2A &other) = delete;

    MaskedSparseAccumulator2A(MaskedSparseAccumulator2A &&other) = delete;

    MaskedSparseAccumulator2A &operator=(const MaskedSparseAccumulator2A &) = delete;

    MaskedSparseAccumulator2A &operator=(MaskedSparseAccumulator2A &&) = delete;

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
//            return std::max(_dirty, _maxIndex * sizeof(ValueT));
            return _dirty;
        } else {
            return _dirty;
        }
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
        assert(_storage._states[key] == NOT_ALLOWED);

        _storage._states[key] = ALLOWED;
    }

    void setNotAllowed(KeyT key) {
        assert(0 <= key && key < _maxIndex);
        assert(_storage._states[key] == ALLOWED);

        _storage._states[key] = NOT_ALLOWED;
    }

    void setInitialized(KeyT key) {
        assert(0 <= key && key < _maxIndex);
        assert(_storage._states[key] == ALLOWED);

        _storage._states[key] = INITIALIZED;
        _dirtyIndices.push_back(key);
    }

    bool erase(T idx) const {
        assert(0 <= idx && idx < _maxIndex);

        if (_storage._states[idx] == DEFAULT_STATE) { return false; }
        _storage._states[idx] = DEFAULT_STATE;
        return true;
    }

    void clear(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        if (_storage._states[idx] != DEFAULT_STATE) {
            _storage._states[idx] = DEFAULT_STATE;
            if constexpr (HAS_VALUE) { memset(&_storage._values[idx], 0xFF, sizeof(ValueT)); }
        }
    }

    void resetStates() {
        for (const auto idx : _dirtyIndices) { clear(idx); }
        _dirtyIndices.clear();
    }

    template<class Iter>
    void resetStates(Iter first, Iter last) {
        for (; first != last; ++first) { clear(*first); }
        resetStates();
    }

    template<bool Sorted>
    void gather(KeyT *&keyIter, ValueT *&valueIter) {
        if (Sorted) { std::sort(_dirtyIndices.begin(), _dirtyIndices.end()); }

        for (const auto idx : _dirtyIndices) {
            *(keyIter++) = idx;
            *(valueIter++) = _storage._values[idx];
            clear(idx);
        }
        _dirtyIndices.clear();
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

#endif //MASKED_SPGEMM_SPARSE_ACCUMULATOR_2A_H
