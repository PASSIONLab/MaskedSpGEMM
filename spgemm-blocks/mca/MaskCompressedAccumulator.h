#ifndef MASKED_SPGEMM_MASK_COMPRESSED_ACCUMULATOR_H
#define MASKED_SPGEMM_MASK_COMPRESSED_ACCUMULATOR_H

#include <cassert>


template<class S, class V>
struct MCAEntry {
    S state;
    V value;
};

template<class S>
struct MCAEntry<S, void> {
    S state;
};

template<class KeyT, class ValueT>
class MaskCompressedAccumulator {
private:
    using T = std::make_unsigned_t<KeyT>;
    using StateT = uint8_t;

    using EntryT = MCAEntry<StateT, ValueT>;

public:
    inline static const StateT EMPTY = std::numeric_limits<StateT>::max();
    inline static const StateT ALLOWED = 0;
    inline static const StateT INITIALIZED = 1;

protected:
    const T _maxIndex;
    EntryT *_entries;

    size_t _dirty;

public:
    MaskCompressedAccumulator(T maxIndex) : _maxIndex(maxIndex) {}

    MaskCompressedAccumulator(const MaskCompressedAccumulator &other) = delete;

    MaskCompressedAccumulator(MaskCompressedAccumulator &&other) = delete;

    MaskCompressedAccumulator &operator=(const MaskCompressedAccumulator &) = delete;

    MaskCompressedAccumulator &operator=(MaskCompressedAccumulator &&) = delete;

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        return {_maxIndex * sizeof(EntryT), sizeof(EntryT)};
    }

    void setBuffer(std::byte *buffer, size_t bufferSize, size_t dirty) {
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
        clearAll(_maxIndex);
    }

    void clearAll(size_t nelems) {
        memset(_entries, 0xFF, nelems * sizeof(EntryT));
    }
};

#endif //MASKED_SPGEMM_MASK_COMPRESSED_ACCUMULATOR_H
