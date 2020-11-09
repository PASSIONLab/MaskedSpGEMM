/*
 * Copyright (c) 2011 Fuji, Goro (gfx) <gfuji@cpan.org>.
 * Copyright (c) 2019 Morwenn.
 *
 * SPDX-License-Identifier: MIT
 */
#include <algorithm>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>
#include <catch.hpp>
#include <gfx/timsort.hpp>

////////////////////////////////////////////////////////////
// Move-only type for benchmarks
//
// std::sort and std::stable_sort are supposed to be able to
// sort collections of types that are move-only and that are
// not default-constructible. The class template move_only
// wraps such a type and can be fed to algorithms to check
// whether they still compile.
//
// Additionally, move_only detects attempts to read the value
// after a move has been performed and throws an exceptions
// when it happens.
//
// It also checks that no self-move is performed, since the
// standard algorithms can't rely on that to work either.
//

namespace
{
    template <typename T> struct move_only {
        // Not default-constructible
        move_only() = delete;

        // Move-only
        move_only(const move_only &) = delete;
        move_only& operator=(const move_only &) = delete;

        // Can be constructed from a T for convenience
        move_only(const T &value) : can_read(true), value(value) {
        }

        // Move operators

        move_only(move_only &&other) : can_read(true), value(std::move(other.value)) {
            if (!exchange(other.can_read, false)) {
                throw std::logic_error("illegal read from a moved-from value");
            }
        }

        auto operator=(move_only &&other) -> move_only & {
            // Self-move should be ok if the object is already in a moved-from
            // state because it incurs no data loss, but should otherwise be
            // frowned upon
            if (&other == this && can_read) {
                throw std::logic_error("illegal self-move was performed");
            }

            // Assign before overwriting other.can_read
            can_read = other.can_read;
            value = std::move(other.value);

            // If the two objects are not the same and we try to read from an
            // object in a moved-from state, then it's a hard error because
            // data might be lost
            if (!exchange(other.can_read, false) && &other != this) {
                throw std::logic_error("illegal read from a moved-from value");
            }

            return *this;
        }

        // A C++11 backport of std::exchange()
        template <typename U> auto exchange(U &obj, U &&new_val) -> U {
            U old_val = std::move(obj);
            obj = std::forward<U>(new_val);
            return old_val;
        }

        // Whether the value can be read
        bool can_read = false;
        // Actual value
        T value;
    };
}

template <typename T>
bool operator<(const move_only<T> &lhs, const move_only<T> &rhs)
{
    return lhs.value < rhs.value;
}

template<typename T>
void swap(move_only<T> &lhs, move_only<T> &rhs)
{
    // This function matters because we want to prevent self-moves
    // but we don't want to prevent self-swaps because it is the
    // responsibility of class authors to make sure that self-swap
    // does the right thing, and not the responsibility of algorithm
    // authors to prevent them from happening

    // Both operands need to be readable
    if (!(lhs.can_read || rhs.can_read)) {
        throw std::logic_error("illegal read from a moved-from value");
    }

    // Swapping the values is enough to preserve the preconditions
    using std::swap;
    swap(lhs.value, rhs.value);
}

TEST_CASE( "shuffle10k_for_move_only_types" ) {
    const int size = 1024 * 10; // should be even number of elements

    std::vector<move_only<int> > a;
    for (int i = 0; i < size; ++i) {
        a.push_back((i + 1) * 10);
    }

    for (int n = 0; n < 100; ++n) {
        std::random_shuffle(a.begin(), a.end());

        gfx::timsort(a.begin(), a.end(), [](const move_only<int> &x, const move_only<int> &y) { return x.value < y.value; });

        for (int i = 0; i < size; ++i) {
            CHECK(a[i].value == (i + 1) * 10);
        }
    }
}

TEST_CASE( "issue14" ) {
    int a[] = {15, 7,  16, 20, 25, 28, 13, 27, 34, 24, 19, 1, 6,  30, 32, 29, 10, 9,
               3,  31, 21, 26, 8,  2,  22, 14, 4,  12, 5,  0, 23, 33, 11, 17, 18};
    std::deque<int> c(std::begin(a), std::end(a));

    gfx::timsort(std::begin(c), std::end(c));
    CHECK(std::is_sorted(std::begin(c), std::end(c)));
}
