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

    EntryT *_entries;
    IT _maxSize;
    IT _size;

public:
    Heap(IT maxSize) : _maxSize(maxSize), _size(0) {}

    Heap(const Heap &other) = delete;

    Heap(Heap &&other) = delete;

    Heap &operator=(const Heap &) = delete;

    Heap &operator=(Heap &&) = delete;

    [[nodiscard]] std::tuple<size_t, size_t> getMemoryRequirement() {
        return {_maxSize * sizeof(EntryT), sizeof(EntryT)};
    }

    void setBuffer(std::byte *buffer, size_t bufferSize, size_t dirty) {
        assert(isAligned(buffer, sizeof(EntryT)));
        assert(_maxSize * sizeof(EntryT) <= bufferSize);
        _entries = reinterpret_cast<EntryT *>(buffer);
        _size = 0;
    }

    void releaseBuffer(size_t &dirty) {
        _entries = nullptr;
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
