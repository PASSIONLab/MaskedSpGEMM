#include <random>
#include <iomanip>
#include "../spgemm-blocks/scan.h"
#include "input-helpers.h"


template<class T, bool inclusive, bool verify = true>
void test(void (*f)(const T *, std::size_t, T *, int), size_t size, size_t niter, size_t numThreads) {
    std::mt19937 gen;
    auto dist = createDistribution<T>(0, 10);

    auto input = new T[size];
    auto output = new T[size];
    auto output_correct = new T[size];

    for (size_t iter = 0; iter < niter; ++iter) {
        for (size_t i = 0; i < size; ++i) { input[i] = dist(gen); }

        f(input, size, output, numThreads);

        if (verify) {
            if (inclusive) {
                output_correct[0] = input[0];
                for (size_t i = 1; i < size; ++i) { output_correct[i] = output_correct[i - 1] + input[i]; }
            } else {
                output_correct[0] = 0;
                for (size_t i = 1; i < size; ++i) { output_correct[i] = output_correct[i - 1] + input[i - 1]; }
            }

            if (size < 32) {
                for (size_t i = 0; i < size; ++i) { std::cout << std::setw(3) << input[i] << " "; }
                std::cout << std::endl;
                for (size_t i = 0; i < size; ++i) { std::cout << std::setw(3) << output[i] << " "; }
                std::cout << std::endl;
                for (size_t i = 0; i < size; ++i) { std::cout << std::setw(3) << output_correct[i] << " "; }
                std::cout << std::endl;
                std::cout << std::endl;
            }

            for (size_t i = 0; i < size; ++i) {
                if (output[i] != output_correct[i]) {
                    std::cerr << "Error: " << i << std::endl;
                    break;
                }
            }
        }
    }

    delete[] input;
    delete[] output;
    delete[] output_correct;
}

int main() {
    for (size_t i = 1; i < 20; ++i) {
        auto size = 1u << i;
        std::cout << "Testing 2^" << i << " (" << size << ") elements..." << std::endl;
        test<int, true>(inclusiveScan, size, 10, 12);
        test<int, false>(exclusiveScan, size, 10, 12);
    }

}