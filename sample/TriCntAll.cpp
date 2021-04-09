#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>
#include <iomanip>


//#include "overridenew.h"
#include "../utility.h"
#include "../CSC.h"
#include "../CSR.h"
#include "../BIN.h"
#include "../hash_mult_hw.h"
#include "../mask_hash_mult.h"
#include "../inner_mult.h"
#include "../heap_mult_generic.h"
#include "sample_common.hpp"
#include "../spa_mult.h"
#include "../spgemm-blocks/masked-spgemm.h"
#include "../spgemm-blocks/masked-spgemm-prof.h"

using namespace std;

static uint16_t calculateChecksum(uint8_t *data, size_t length) {
    uint32_t checksum = 0;
    auto *data16 = (uint16_t *) data;
    auto length16 = length >> (size_t) 1;

    while (length16--) {
        checksum += *data16;
        data16++;
        if (checksum & 0xFFFF0000) {
            checksum &= 0xFFFF;
            checksum++;
        }
    }

    // If number of bytes is odd, add remaining byte
    if (length & 0x1) {
        checksum += *((uint8_t *) data16);
        if (checksum & 0xFFFF0000) {
            checksum &= 0xFFFF;
            checksum++;
        }
    }

    return (uint16_t) ~(checksum & 0xFFFF);
}

template<class IT, class NT, template<class, class> class AT>
std::string checksum(const AT<IT, NT> &A) {
    uint16_t valuesCSC = calculateChecksum(reinterpret_cast<uint8_t *>(A.values), A.nnz * sizeof(NT));
    uint16_t idsCSC;
    if constexpr (std::is_same<AT<IT, NT>, CSR<IT, NT>>::value) {
        idsCSC = calculateChecksum(reinterpret_cast<uint8_t *>(A.colids), A.nnz * sizeof(IT));
    } else {
        idsCSC = calculateChecksum(reinterpret_cast<uint8_t *>(A.rowids), A.nnz * sizeof(IT));
    }

    return to_string(valuesCSC) + "|" + to_string(idsCSC);
}

static const char *getFileName(const char *path) {
    int len = strlen(path);
    while (path[len] != '/' && len >= 0) { len--; }
    return path + len + 1;
}

const char *fileName;

template<class IT, class NT,
        template<class, class> class AT,
        template<class, class> class BT,
        template<class, class> class CT = AT,
        template<class, class> class MT>
void run(const std::string &name,
         void(*f)(const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &, const MT<IT, NT> &,
                  multiplies<NT>, plus<NT>, unsigned),
         size_t niters, vector<int> &tnums, size_t nfop,
         const AT<IT, NT> &A, const BT<IT, NT> &B, const MT<IT, NT> &M) {
    cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): "
         << nfop << endl << endl;

    for (int tnum : tnums) {
        omp_set_num_threads(tnum); // TODO: update get_flop to use numThreads methods and remove this

        CT<IT, NT> C;

        // The first iteraion is excluded from evaluation if there is only one iteration
        if (niters != 1) { f(A, B, C, M, multiplies<NT>(), plus<NT>(), tnum); }

        double ave_msec = 0;
        for (int i = 0; i < niters; ++i) {
            C.make_empty();

            double start = omp_get_wtime();
            f(A, B, C, M, multiplies<NT>(), plus<NT>(), tnum);
            double end = omp_get_wtime();

            double msec = (end - start) * 1000;
            ave_msec += msec;
        }

        ave_msec /= niters;
        double mflops = (double) nfop / ave_msec / 1000;

        std::cout << name << " returned with " << C.nnz << " nonzeros. "
                  << "Compression ration is " << ((float) nfop / 2) / (float) (C.nnz) << std::endl;
        std::cout << name << " with " << std::setw(3) << tnum << " threads computes C = A * B in "
                  << ave_msec << " [milli seconds] (" << mflops << " [MFLOPS])" << std::endl;

        std::cout << "Checksum: " << checksum(C) << std::endl;

        std::cout << "LOG," << fileName << "," << name << "," << typeid(IT).name() << "|" << typeid(NT).name()
                  << "," << tnum << "," << ave_msec << "," << mflops << ","
                  << C.nnz << "," << C.sumall() << "," << checksum(C) << std::endl;

        std::cout << std::endl;
        C.make_empty();
    }
}

template<class IT, class NT>
void process(CSC<IT, NT> &A) {
    std::vector<std::pair<IT, IT>> rowcnts(A.rows, std::pair<IT, IT>{0, 0});

    for (IT i = 0; i < A.rows; i++) { rowcnts[i].second = i; }

    for (IT i = 0; i < A.nnz; i++) { rowcnts[A.rowids[i]].first++; }

    std::sort(rowcnts.begin(), rowcnts.end(), std::greater<>{});

    auto triples = new Triple<IT, NT>[A.nnz];

    IT idx = 0;
    for (IT i = 0; i < A.cols; i++) {
        for (IT j = A.colptr[i]; j < A.colptr[i + 1]; j++) {
            IT col = rowcnts[i].second;
            IT row = rowcnts[A.rowids[j]].second;
            if (col < row) {
                triples[idx] = Triple(row, col, NT(1));
                idx++;
            } else if (col > row) {
                triples[idx] = Triple(col, row, NT(1));
                idx++;
            }
        }
    }

    std::sort(triples, triples + idx, [](auto lhs, auto rhs) {
        return rhs.row < lhs.row || (rhs.row == lhs.row && rhs.col < lhs.col);
    });

    IT dst = 1;
    for (IT src = 1; src < idx; src++) {
        if (triples[src].row != triples[src - 1].row || triples[src].col != triples[src - 1].col) {
            triples[dst] = triples[src];
            dst++;
        }
    }
    idx = dst;

    A = CSC<IT, NT>{triples, idx, A.rows, A.cols};

    delete[] triples;
}

int main(int argc, char *argv[]) {
    using Value_t = long unsigned;
    using Index_t = long;

    vector<int> tnums;
    if (argc < 3) {
        cout << "Normal usage: ./tricnt-all matrix1.mtx <numthreads>" << endl;
        return -1;
    } else {
        cout << "Running on " << argv[2] << " processors" << endl << endl;
        tnums = {atoi(argv[2])};
    }

    fileName = getFileName(argv[1]);
    string inputname1 = argv[1];
    CSC<Index_t, Value_t> A_csc;
    ReadASCII(inputname1, A_csc);
    process(A_csc);
    CSR<Index_t, Value_t> A_csr(A_csc); //converts, allocates and populates

    size_t innerIters = std::getenv("INNER_ITERS") ? std::stoul(std::getenv("INNER_ITERS")) : 1;
    size_t outerIters = std::getenv("OUTER_ITERS") ? std::stoul(std::getenv("OUTER_ITERS")) : 1;

    std::cout << "Iters: " << outerIters << " x " << innerIters << std::endl;

    std::cout << std::endl;

    std::size_t flop = get_flop(A_csc, A_csc);
    for (size_t i = 0; i < outerIters; i++) {
        run("MaskedSpGEMM1p",         MaskedSpGEMM1p_prof<MaskedSPA2A>,         innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM1p<MaskedSPA2A>",        MaskedSpGEMM1p<MaskedSPA2A>,        innerIters, tnums, flop, A_csr, A_csr, A_csr);
        continue;
        // @formatter:off
        run("innerSpGEMM_nohash<false-false>", innerSpGEMM_nohash<false, false>,   innerIters, tnums, flop, A_csr, A_csc, A_csr);
        std::cout << "LOG,separator" << std::endl;

        run("mxm_hash_mask",                      mxm_hash_mask,                      innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("mxm_hash_mask_wobin",                mxm_hash_mask_wobin,                innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM2p<MaskedHash>",         MaskedSpGEMM2p<MaskedHash>,         innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM1p<MaskedHash>",         MaskedSpGEMM1p<MaskedHash>,         innerIters, tnums, flop, A_csr, A_csr, A_csr);
        std::cout << "LOG,separator" << std::endl;

        run("MaskedSPASpGEMM",                    MaskedSPASpGEMM,                    innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM2p<MaskedSPA>",          MaskedSpGEMM2p<MaskedSPA>,          innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM1p<MaskedSPA>",          MaskedSpGEMM1p<MaskedSPA>,          innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM2p<MaskedSPA2A>",        MaskedSpGEMM2p<MaskedSPA2A>,        innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM1p<MaskedSPA2A>",        MaskedSpGEMM1p<MaskedSPA2A>,        innerIters, tnums, flop, A_csr, A_csr, A_csr);
        std::cout << "LOG,separator" << std::endl;

        run("HeapSpGEMM<rowAlg::MaskIndexed_v1>", HeapSpGEMM<rowAlg::MaskIndexed_v1>, innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("HeapSpGEMM<rowAlg::MaskIndexed_v2>", HeapSpGEMM<rowAlg::MaskIndexed_v2>, innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("HeapSpGEMM<rowAlg::MaskIndexed_v3>", HeapSpGEMM<rowAlg::MaskIndexed_v3>, innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM2p<MaskIndexed>",        MaskedSpGEMM2p<MaskIndexed>,        innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM1p<MaskIndexed>",        MaskedSpGEMM1p<MaskIndexed>,        innerIters, tnums, flop, A_csr, A_csr, A_csr);
        std::cout << "LOG,separator" << std::endl;

        run("HeapSpGEMM<rowAlg::MaskedHeap_v0>", HeapSpGEMM<rowAlg::MaskedHeap_v0>,   innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("HeapSpGEMM<rowAlg::MaskedHeap_v1>", HeapSpGEMM<rowAlg::MaskedHeap_v1>,   innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("HeapSpGEMM<rowAlg::MaskedHeap_v2>", HeapSpGEMM<rowAlg::MaskedHeap_v2>,   innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM2p<MaskedHeap_v1>",     MaskedSpGEMM2p<MaskedHeap_v1>,       innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM1p<MaskedHeap_v1>",     MaskedSpGEMM1p<MaskedHeap_v1>,       innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM2p<MaskedHeap_v2>",     MaskedSpGEMM2p<MaskedHeap_v2>,       innerIters, tnums, flop, A_csr, A_csr, A_csr);
        run("MaskedSpGEMM1p<MaskedHeap_v2>",     MaskedSpGEMM1p<MaskedHeap_v2>,       innerIters, tnums, flop, A_csr, A_csr, A_csr);
        // @formatter:on
    }

    size_t nrows = std::min<size_t>(A_csr.rows, 1000);
    for (int i = 0; i < nrows; i++) {
        auto nnz = A_csr.rowptr[i + 1] - A_csr.rowptr[i];
        std::cout << i << "," << nnz;
        for (auto j = 1; j < times.size(); j++) {
            std::cout << "," << times[j][i];
        }
        std::cout << std::endl;
    }

    A_csc.make_empty();
    A_csr.make_empty();

    return 0;
}
