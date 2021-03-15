#ifndef MASKEDS_PGEMM_HEAP_H
#define MASKEDS_PGEMM_HEAP_H

#include <algorithm>

template<class IT>
class Heap {
    struct EntryT {
        IT key;
        IT runr;
        IT loc; // location of the current nonzero that column of A (absolute)

        // Operators are swapped for performance
        // If you want/need to convert them back to their normal definitions, don't
        // forget to add "greater< HeapEntry<T> >()" optional parameter to all the
        // heap operations operating on HeapEntry<T> objects. For example:
        // push_heap(heap, heap + kisect, greater< HeapEntry<T> >());

        bool operator>(const EntryT &rhs) const { return (key < rhs.key); }

        bool operator<(const EntryT &rhs) const { return (key > rhs.key); }

        bool operator==(const EntryT &rhs) const { return (key == rhs.key); }
    };

    const IT _capacity;
    IT _size;
    EntryT *_entries;

    size_t _dirty;
    size_t _maxSize;

public:
    Heap(IT capacity) : _capacity(capacity), _size(0), _entries(nullptr), _dirty(0), _maxSize(0) {}

    Heap(const Heap &other) = delete;

    Heap(Heap &&other) = delete;

    Heap &operator=(const Heap &) = delete;

    Heap &operator=(Heap &&) = delete;

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        return {_capacity * sizeof(EntryT), sizeof(EntryT)};
    }

    void setBuffer(std::byte *buffer, size_t bufferSize, size_t dirty) {
        assert(isAligned(buffer, sizeof(EntryT)));
        assert(_capacity * sizeof(EntryT) <= bufferSize);

        _size = 0;
        _dirty = dirty;
        _entries = reinterpret_cast<EntryT *>(buffer);
    }

    [[nodiscard]] size_t releaseBuffer() {
#if defined(DEBUG)
        assert(_size == 0);
        _entries = nullptr;
#endif
        return std::max(_dirty, _maxSize * sizeof(EntryT));
    }

    void append(IT key, IT runr, IT loc) {
        _entries[_size].key = key;
        _entries[_size].runr = runr;
        _entries[_size].loc = loc;
        ++_size;
    }

    [[nodiscard]] bool isEmpty() {
        return _size == 0;
    }

    [[nodiscard]] EntryT& top() {
        return _entries[0];
    }

    void make() {
        if (_size > _maxSize) { _maxSize = _size; }
        std::make_heap(_entries, _entries + _size);
    }

    void pop() {
        std::pop_heap(_entries, _entries + _size);
        --_size;
    }

    void sinkRoot() {
        std::pop_heap(_entries, _entries + _size);
        std::push_heap(_entries, _entries + _size);
    }

};

#endif //MASKEDS_PGEMM_HEAP_H
