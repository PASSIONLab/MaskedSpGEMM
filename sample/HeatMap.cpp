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
void createMatrices(const std::string &name, IT nrows, IT ncols, IT degMin, IT degMax,
                    std::vector<std::pair<CSC<IT, NT>, IT>> &csc, std::vector<std::pair<CSR<IT, NT>, IT>> &csr) {
    std::cout << "Crating " << name << "s... ";
    std::cout.flush();

    IT numInputs = 0;
    for (IT deg = degMin; deg <= degMax; deg *= 2) { numInputs++; }
    csc.resize(numInputs);
    csr.resize(numInputs);

    IT idx = 0;
    for (IT deg = degMin; deg <= degMax; deg *= 2) {
        Triple<IT, NT> *triples = nullptr;
        auto nvals = generateRandomTriples(nrows, ncols, nrows * deg, triples);
        CSC<IT, NT> matrix(triples, nvals, nrows, ncols);
        delete[] triples;

        csc[idx] = std::make_pair(matrix, deg);
        csr[idx] = std::make_pair(CSR<IT, NT>(matrix), deg);
        ++idx;
    }

    std::cout << "Done" << std::endl;
}

template<template<class, class> class AT, class IT, class NT>
void deleteMatrices(std::vector<std::pair<AT<IT, NT>, IT>> &matrices) {
    for (auto &it : matrices) { it.first.make_empty(); }
}


template<class IT, class NT>
void createMatrices(const std::string &name, IT nrows, IT ncols, IT degMin, IT degMax,
                    std::vector<std::pair<CSR<IT, NT>, IT>> &csr) {
    std::vector<std::pair<CSC<IT, NT>, IT>> csc;
    createMatrices(name, nrows, ncols, degMin, degMax, csc, csr);
    deleteMatrices(csc);
}

int main() {
    using Index_t = uint32_t;
    using Value_t = long unsigned;

    size_t niter = 10, witer = 2, nthreads = 12;
    Index_t dimensionMin = 1024, dAMin = 1, dBMin = 1, dMMin = 1;
    Index_t dimensionMax = 1024 * 1024, dAMax = 128, dBMax = 128, dMMax = 256;
    Index_t maxRowsA = 32 * 1024;
    bool sameAB = true;
    omp_set_num_threads(nthreads);

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

    std::vector<std::pair<std::string,
            void (*)(const CSR<Index_t, Value_t> &, const CSC<Index_t, Value_t> &,
                     CSR<Index_t, Value_t> &, const CSR<Index_t, Value_t> &,
                     multiplies<Value_t>, plus<Value_t>, unsigned)>>
            csrcsc
            {
//                    {"innerSpGEMM_nohash", innerSpGEMM_nohash<false, false>},
            };

    std::cout << "dimension,";
    if (sameAB) { std::cout << "degAB,"; } else { std::cout << "degA," << "debB,"; }
    std::cout << "degM," << "flops";
    for (const auto &it : csrcsc) { std::cout << "," << it.first; }
    for (const auto &it : csrscr) { std::cout << "," << it.first; }
    std::cout << std::endl;

    for (Index_t dim = dimensionMin; dim <= dimensionMax; dim *= 2) {
        std::vector<std::pair<CSR<Index_t, Value_t>, Index_t>> AsCSR, BsCSR, MsCSR;
        std::vector<std::pair<CSC<Index_t, Value_t>, Index_t>> BsCSC;
        createMatrices("A", std::min(maxRowsA, dim), dim, dAMin, dAMax, AsCSR);
        createMatrices("B", dim, dim, dBMin, dBMax, BsCSC, BsCSR);
        createMatrices("M", std::min(maxRowsA, dim), dim, dMMin, dMMax, MsCSR);


        for (int ia = 0; ia < AsCSR.size(); ia++) {
            for (int ib = 0; ib < BsCSR.size(); ib++) {
                if (sameAB) { ib = ia; }
                for (int im = 0; im < MsCSR.size(); im++) {
                    auto flops = calculateFlops(AsCSR[ia].first, BsCSR[ib].first, flopsPerRow, 12);
                    std::cout << dim << ",";
                    if (sameAB) { std::cout << AsCSR[ia].second << ", "; }
                    else { std::cout << AsCSR[ia].second << "," << BsCSR[ib].second << ","; }
                    std::cout << MsCSR[im].second << "," << flops;

                    for (const auto &alg : csrcsc) {
                        auto time = run(alg.second, AsCSR[ia].first, BsCSC[ib].first, MsCSR[im].first,
                                        witer, niter, nthreads);

                        std::cout << "," << time;
                    }

                    for (const auto &alg : csrscr) {
                        auto time = run(alg.second, AsCSR[ia].first, BsCSR[ib].first, MsCSR[im].first,
                                        witer, niter, nthreads);
                        std::cout << "," << time;
                    }
                    std::cout << std::endl;
                }
                if (sameAB) { break; }
            }
        }

        deleteMatrices(AsCSR);
        deleteMatrices(BsCSR);
        deleteMatrices(BsCSC);
        deleteMatrices(MsCSR);
    }

    delete[] flopsPerRow;
}