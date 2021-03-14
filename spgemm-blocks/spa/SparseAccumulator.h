#ifndef MASKED_SPGEMM_SPARSE_ACCUMULATOR_H
#define MASKED_SPGEMM_SPARSE_ACCUMULATOR_H

#include <utility>
#include <limits>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "../util.h"

template<class Key, class EntryT>
class SparseAccumulatorBase {
public:
    using StateT = decltype(std::declval<EntryT>().state);
    using T = std::make_unsigned_t<Key>;

    static_assert(std::is_unsigned_v<StateT> && !std::is_same_v<StateT, bool>);

public:
    inline static const Key EMPTY = std::numeric_limits<StateT>::max();

protected:
    const T _maxIndex;    // Used only for debugging
    const T _maxEntries;    // Used only for debugging
    EntryT *_entries;
    T _numEntriesIndices;
    T *_entryIndices;
    size_t &_dirtyMemSize;

public:
    SparseAccumulatorBase(T maxIndex, size_t maxEntries, std::byte *buffer, size_t bufferSize, size_t &dirtyMemSize)
            : _maxIndex(maxIndex), _maxEntries(maxEntries), _numEntriesIndices(0), _dirtyMemSize(dirtyMemSize) {
        size_t cleaned = splitMemory(buffer, bufferSize, dirtyMemSize, _entryIndices, maxEntries, _entries, maxIndex);
        dirtyMemSize -= cleaned;
    }

    SparseAccumulatorBase(const SparseAccumulatorBase &other) = delete;

    SparseAccumulatorBase(SparseAccumulatorBase &&other) = delete;

    SparseAccumulatorBase &operator=(const SparseAccumulatorBase &) = delete;

    SparseAccumulatorBase &operator=(SparseAccumulatorBase &&) = delete;

    ~SparseAccumulatorBase() {
        _dirtyMemSize = std::max(roundUp<sizeof(T)>(_numEntriesIndices), _dirtyMemSize);
    }

    [[nodiscard]] static size_t requiredMemory(size_t maxIndex, size_t maxEntries) {
        // Memory required for the entryIndices + memoryRequired for the entries
        return roundUp<sizeof(EntryT)>(maxIndex) * sizeof(T) + maxEntries * sizeof(EntryT);
    }

    [[nodiscard]] constexpr static size_t requiredAlignment() {
        return std::lcm(sizeof(T), sizeof(EntryT));
    }

    [[nodiscard]] EntryT &operator[](size_t key) {
        assert(0 <= key && key < _maxIndex);
        return _entries[key];
    }

    bool erase(Key key) const {
        bool erased = _entries[key].state != EMPTY;
        _entries[key].state = EMPTY;
        return erased;
    }

    bool reset() {
        for (size_t i = 0; i < _numEntriesIndices; i++) {
            memset(&(_entries[_entryIndices[i]]), 0xFF, sizeof(EntryT));
        }
    }

protected:
    void addEntry(Key key) {
        _entryIndices[_numEntriesIndices++] = key;
    }
};


template<class...>
struct SPAEntry;

// region SPA for symbolic phase

template<class T>
struct SPAEntry<T> {
    T state;
};

template<class Key>
class SparseAccumulator : public SparseAccumulatorBase<Key, SPAEntry<uint8_t>> {
    using super = SparseAccumulatorBase<Key, SPAEntry<uint8_t>>;

public:
    SparseAccumulator(typename super::T maxIndex, size_t maxEntries,
                      std::byte *buffer, size_t bufferSize, size_t &dirtyMemSize)
            : super(maxIndex, maxEntries, buffer, bufferSize, dirtyMemSize) {}

    bool insert(Key key) {
        assert(0 <= key && key < this->_maxIndex);
        if (this->_entries[key].state != super::EMPTY) { return false; }

        this->_entries[key].state = 0;
        return true;
    }
};

//endregion

// region SPA for numeric phae

// endregion

#endif //MASKED_SPGEMM_SPARSE_ACCUMULATOR_H
