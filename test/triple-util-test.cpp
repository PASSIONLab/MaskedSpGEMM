#include <iomanip>
#include "../util/triple-util.h"

int main() {
    using Index_t = long;
    using Value_t = long unsigned;

    Triple<Index_t, Value_t> *triples;

    auto nvals = generateRandomTriples<Index_t, Value_t>(10, 10, 30, triples);
    for (size_t i = 0; i < nvals; i++) {
        const auto &t = triples[i];
        std::cout << std::setw(3) << i << ": (" << t.row << ", " << t.col << ") - " << t.val << std::endl;
    }


    return 0;
}

