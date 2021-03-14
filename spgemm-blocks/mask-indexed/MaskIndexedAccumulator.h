#ifndef MASKED_SPGEMM_MASK_INDEXED_ACCUMULATOR_H
#define MASKED_SPGEMM_MASK_INDEXED_ACCUMULATOR_H

#include <cassert>


template<class S, class V>
struct MaskIndexedEntry {
    S state;
    V value;
};

template<class S>
struct MaskIndexedEntry<S, void> {
    S state;
};

template<class KeyT, class ValueT>
class MaskIndexedAccumulator {
private:
    using T = std::make_unsigned_t<KeyT>;
    using StateT = uint8_t;

    using EntryT = MaskIndexedEntry<StateT, ValueT>;

public:
    inline static const StateT EMPTY = std::numeric_limits<StateT>::max();
    inline static const StateT ALLOWED = 0;
    inline static const StateT INITIALIZED = 1;

protected:
    const T _maxIndex;
    EntryT *_entries;

public:
    MaskIndexedAccumulator(T maxIndex) : _maxIndex(maxIndex) {}

    MaskIndexedAccumulator(const MaskIndexedAccumulator &other) = delete;

    MaskIndexedAccumulator(MaskIndexedAccumulator &&other) = delete;

    MaskIndexedAccumulator &operator=(const MaskIndexedAccumulator &) = delete;

    MaskIndexedAccumulator &operator=(MaskIndexedAccumulator &&) = delete;

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        return {_maxIndex * sizeof(EntryT), sizeof(EntryT)};
    }

    void setBuffer(std::byte *buffer, size_t bufferSize, size_t dirty) {
        assert(isAligned(buffer, sizeof(EntryT)));
        assert(_maxIndex * sizeof(EntryT) <= bufferSize);
        // TODO: maybe use memorySplit
        _entries = reinterpret_cast<EntryT *>(buffer);
        clearAll();
    }

    void releaseBuffer(size_t &dirty) {
        _entries = nullptr;
    }

    [[nodiscard]] bool isEmpty(T idx) const {
        assert(0 <= idx && idx < _maxIndex);
        return _entries[idx].state == EMPTY;
    }

    [[nodiscard]] EntryT &operator[](T idx) {
        assert(0 <= idx && idx < _maxIndex);
        return _entries[idx];
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
            if constexpr (std::is_same_v<ValueT, void>) {
                memset(&_entries[idx].value, 0xFF, sizeof(ValueT));
            }
        }
    }

    void clearAll() {
        memset(_entries, 0xFF, _maxIndex * sizeof(EntryT));
    }
};

#endif //MASKED_SPGEMM_MASK_INDEXED_ACCUMULATOR_H
