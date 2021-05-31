#include <cstddef>
#include <cstdint>
#include <iomanip>
#include "../util/triple-util.h"
#include "../CSC.h"
#include "../CSR.h"
#include "../spgemm-blocks/masked-spgemm.h"
#include "../spgemm-blocks/common.h"
#include "../inner_mult.h"

template<class IT, class NT,
        template<class, class> class AT,
        template<class, class> class BT,
        template<class, class> class CT = AT,
        template<class, class> class MT>
double run(void(*f)(const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &, const MT<IT, NT> &,
                    multiplies<NT>, plus<NT>, unsigned),
           AT<IT, NT> &A, BT<IT, NT> &B, MT<IT, NT> &M, size_t witer, size_t niters, size_t nthreads) {
    long totalTime = 0;
    CT<IT, NT> C;

    // Warmup iterations
    for (size_t i = 0; i < witer; i++) {
        C.make_empty();
        f(A, B, C, M, std::multiplies<NT>{}, std::plus<NT>{}, nthreads);
    }

    for (size_t i = 0; i < niters; i++) {
        C.make_empty();
        auto start = std::chrono::high_resolution_clock::now();
        f(A, B, C, M, std::multiplies<NT>{}, std::plus<NT>{}, nthreads);
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        totalTime += time;
//        std::cout << std::setw(8) << time << std::setw(8) << C.nnz << std::endl;
    }

    C.make_empty();

    return double(totalTime) / double(niters) / 1e6;
}

template<class IT, class NT>
void createMatrices(const std::string &name, IT dimension, IT degMin, IT degMax,
                    std::vector<std::pair<CSR<IT, NT>, IT>> &inputs) {
    std::cout << "Crating " << name << "s... ";
    std::cout.flush();

    IT numInputs = 0;
    for (IT deg = degMin; deg <= degMax; deg *= 2) { numInputs++; }
    inputs.resize(numInputs);

    IT idx = 0;
    for (IT deg = degMin; deg <= degMax; deg *= 2) {
        Triple<IT, NT> *triples = nullptr;
        auto nvals = generateRandomTriples(dimension, dimension, dimension * deg, triples);
        CSC<IT, NT> csc(triples, nvals, dimension, dimension);
        delete[] triples;
        inputs[idx++] = std::make_pair(CSR<IT, NT>(csc), deg);
        csc.make_empty();
    }

    std::cout << "Done" << std::endl;
}

template<class IT, class NT>
void deleteMatrices(std::vector<std::pair<CSR<IT, NT>, IT>> &matrices) {
    for (auto &it : matrices) { it.first.make_empty(); }
}

int main() {
    using Index_t = uint32_t;
    using Value_t = long unsigned;

    size_t niter = 25, witer = 3, nthreads = 12;
    Index_t dimensionMin = 128 * 1024, dAMin = 1, dBMin = 1, dMMin = 1;
    Index_t dimensionMax = 128 * 1024, dAMax = 128, dBMax = 128, dMMax = 128;

    auto flopsPerRow = new Index_t[dimensionMax];

    std::vector<std::pair<std::string,
            void (*)(const CSR<Index_t, Value_t> &, const CSR<Index_t, Value_t> &,
                     CSR<Index_t, Value_t> &, const CSR<Index_t, Value_t> &,
                     multiplies<Value_t>, plus<Value_t>, unsigned)>>
            csrscr
            {
                    {"MaskedHash",    MaskedSpGEMM1p<MaskedHash>},
                    {"MSA2A",         MaskedSpGEMM1p<MSA2A>},
                    {"MCA",           MaskedSpGEMM1p<MCA>},
                    {"MaskedHeap_v1", MaskedSpGEMM1p<MaskedHeap_v1>},
                    {"MaskedHeap_v2", MaskedSpGEMM1p<MaskedHeap_v2>},
            };

    for (Index_t dim = dimensionMin; dim <= dimensionMax; dim *= 2) {
        std::vector<std::pair<CSR<Index_t, Value_t>, Index_t>> As, Bs, Ms;
        createMatrices("A", dim, dAMin, dAMax, As);
        createMatrices("B", dim, dBMin, dBMax, Bs);
        createMatrices("M", dim, dMMin, dMMax, Ms);

        std::cout << "degA," << "degM," << "MaskedHash," << "MSA2A," << "MCA," << "MaskedHeap_v1," << "MaskedHeap_v2"
                  << std::endl;

        for (auto &itA : As) {
//            for (auto &itB : Bs) {
            {
                auto &itB = itA;
                for (auto &itM : Ms) {
                    auto flops = calculateFlops(itA.first, itB.first, flopsPerRow, 12);
//                    std::cout << dim << " " << itA.second << " " << itB.second << " "
//                              << itM.second << " " << flops << std::endl;

                    std::cout << itA.second << "," << itM.second;
                    for (const auto &alg : csrscr) {
                        std::cout << "," << run(alg.second, itA.first, itB.first, itM.first, witer, niter, nthreads);;
                    }
                    std::cout << std::endl;
                }
            }
        }

        deleteMatrices(As);
        deleteMatrices(Bs);
        deleteMatrices(Ms);
    }

    delete[] flopsPerRow;
}