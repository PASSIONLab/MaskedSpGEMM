#ifndef MASKED_SPGEMM_SPARSE_ACCUMULATOR_1A_H
#define MASKED_SPGEMM_SPARSE_ACCUMULATOR_1A_H

#include <utility>
#include <limits>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>
#include <vector>

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

template<class KeyT, class ValueT, bool Complemented = false>
class MaskedSparseAccumulator1A {
protected:
    using T = std::make_unsigned_t<KeyT>;
    using StateT = uint8_t;

    using EntryT = SPAEntry<StateT, ValueT>;

public:
    inline static const bool HAS_VALUE = !std::is_same_v<ValueT, void>;

    // Initial value (default value) for the states is 0b11....11.
    inline static const StateT NOT_ALLOWED = !Complemented ? std::numeric_limits<StateT>::max() : 0;
    inline static const StateT ALLOWED = !Complemented ? 0 : std::numeric_limits<StateT>::max();
    inline static const StateT INITIALIZED = 1;

    inline static const StateT DEFAULT_STATE = !Complemented ? NOT_ALLOWED : ALLOWED;

protected:
    const T _maxIndex;
    EntryT *_entries;

    size_t _dirty;
    std::vector<KeyT> _dirtyIndices; // TODO: replace with an array (and remove vector include)

public:
    MaskedSparseAccumulator1A(T maxIndex) : _maxIndex(maxIndex) {}

    MaskedSparseAccumulator1A(const MaskedSparseAccumulator1A &other) = delete;

    MaskedSparseAccumulator1A(MaskedSparseAccumulator1A &&other) = delete;

    MaskedSparseAccumulator1A &operator=(const MaskedSparseAccumulator1A &) = delete;

    MaskedSparseAccumulator1A &operator=(MaskedSparseAccumulator1A &&) = delete;

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

    StateT &getState(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        return _entries[idx].state;
    }

    template<class U = ValueT>
    std::enable_if_t<!std::is_same_v<U, void>, ValueT> &getValue(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        return _entries[idx].value;
    }

    [[nodiscard]] EntryT &operator[](T idx) {
        assert(0 <= idx && idx < _maxIndex);

        return _entries[idx];
    }

    void setAllowed(KeyT key) {
        assert(0 <= key && key < _maxIndex);
        assert(_entries[key].state == NOT_ALLOWED);

        _entries[key].state = ALLOWED;
    }

    void setNotAllowed(KeyT key) {
        assert(0 <= key && key < _maxIndex);
        assert(_entries[key].state == ALLOWED);

        _entries[key].state = NOT_ALLOWED;
    }

    void setInitialized(KeyT key) {
        assert(0 <= key && key < _maxIndex);
        assert(_entries[key].state == ALLOWED);

        _entries[key].state = INITIALIZED;
        _dirtyIndices.push_back(key);
    }

    bool erase(T idx) const {
        assert(0 <= idx && idx < _maxIndex);

        if (_entries[idx].state == DEFAULT_STATE) { return false; }
        _entries[idx].state = DEFAULT_STATE;
        return true;
    }

    void clear(T idx) {
        assert(0 <= idx && idx < _maxIndex);

        if (_entries[idx].state == DEFAULT_STATE) { return; }

        _entries[idx].state = DEFAULT_STATE;
        if constexpr (HAS_VALUE) { memset(&_entries[idx].value, 0xFF, sizeof(ValueT)); }
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
            *(valueIter++) = _entries[idx].value;
            clear(idx);
        }
        _dirtyIndices.clear();
    }

    void clearAll() {
        memset(_entries, 0xFF, _maxIndex * sizeof(EntryT));
    }

    [[nodiscard]] bool isInitialized() const {
        EntryT initialized;
        memset(&initialized, 0xFF, sizeof(EntryT));

        for (size_t i = 0; i < _maxIndex; i++) {
            if (memcmp(&initialized, _entries + i, sizeof(EntryT)) != 0) { return false; }
        }

        return true;
    }
};

#endif //MASKED_SPGEMM_SPARSE_ACCUMULATOR_1A_H
