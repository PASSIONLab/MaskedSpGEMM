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

using namespace std;

#define ITERS 1

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

        std::cout << "LOG," << fileName << "," << name << "," << typeid(IT).name() << "," << typeid(NT).name()
                  << "," << tnum << "," << ave_msec << "," << mflops << "," << C.nnz << "," << checksum(C) << std::endl;

        std::cout << std::endl;
        C.make_empty();
    }
}

int main(int argc, char *argv[]) {
    using Value_t = double;
    using Index_t = int;

    vector<int> tnums;
    if (argc < 4) {
        cout << "Normal usage: ./spgemm matrix1.mtx matrix2.mtx mask.mtx <numthreads>" << endl;
        return -1;
    } else if (argc < 5) {
        cout << "Normal usage: ./spgemm matrix1.mtx matrix2.mtx mask.mtx <numthreads>" << endl;

#ifdef KNL_EXE
        cout << "Running on 68, 136, 204, 272 threads" << endl << endl;
        tnums = {68, 136, 204, 272};
        // tnums = {1, 2, 4, 8, 16, 32, 64, 68, 128, 136, 192, 204, 256, 272}; // for scalability test
#else
        cout << "Running on 4, 8, 16, 32, 64 threads" << endl;
        tnums = {4, 8, 16, 32, 64}; // for hashwell
#endif
    } else {
        cout << "Running on " << argv[4] << " processors" << endl << endl;
        fileName = getFileName(argv[1]);
        tnums = {atoi(argv[4])};
    }

    string inputname1 = argv[1];
    CSC<Index_t, Value_t> A_csc;
    ReadASCII(inputname1, A_csc);
    CSR<Index_t, Value_t> A_csr(A_csc); //converts, allocates and populates
    CSR<Index_t, Value_t> A_csr_t(A_csc, true); //converts, allocates and populates

    string inputname2 = argv[2];
    CSC<Index_t, Value_t> B_csc;
    ReadASCII(inputname2, B_csc);
    CSR<Index_t, Value_t> B_csr(B_csc); //converts, allocates and populates
    CSR<Index_t, Value_t> B_csr_t(B_csc, true); //converts, allocates and populates

    string inputname3 = argv[3];
    CSC<Index_t, Value_t> M_csc;
    ReadASCII(inputname3, M_csc);
    CSR<Index_t, Value_t> M_csr(M_csc); //converts, allocates and populates
    CSR<Index_t, Value_t> M_csr_t(M_csc, true); //converts, allocates and populates

    A_csr.Sorted();
    B_csr.Sorted();
    M_csr.Sorted();

    A_csc.Sorted();
    B_csc.Sorted();
    M_csc.Sorted();

    A_csr_t.Sorted();
    M_csr_t.Sorted();
    M_csr_t.Sorted();

    size_t innerIters = std::getenv("INNER_ITERS") ? std::stoul(std::getenv("INNER_ITERS")) : ITERS;
    size_t outerIters = std::getenv("OUTER_ITERS") ? std::stoul(std::getenv("OUTER_ITERS")) : 1;

    std::cout << "Iters: " << outerIters << " x " << innerIters << std::endl;

    std::cout << std::endl;

    std::size_t flop = get_flop(A_csc, B_csc);
    for (size_t i = 0; i < outerIters; i++) {
        // @formatter:off
        run("MaskedSPASpGEMM CSR", MaskedSPASpGEMM, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("innerSpGEMM_nohash<false-false> CSR/CSC", innerSpGEMM_nohash<false, false>, innerIters, tnums, flop, A_csr, B_csc, M_csr);
        run("mxm_hash_mask CSR", mxm_hash_mask, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("mxm_hash_mask_wobin CSR", mxm_hash_mask_wobin, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("HeapSpGEMM<rowAlg::MaskedBasicHeap_v1> CSR", HeapSpGEMM<rowAlg::MaskedHeap_v0>, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("HeapSpGEMM<rowAlg::MaskedBasicHeap_v2> CSR", HeapSpGEMM<rowAlg::MaskedHeap_v1>, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("HeapSpGEMM<rowAlg::MaskedBasicHeap_v3> CSR", HeapSpGEMM<rowAlg::MaskedHeap_v2>, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("HeapSpGEMM<rowAlg::MaskIndexed_v1> CSR", HeapSpGEMM<rowAlg::MaskIndexed_v1>, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("HeapSpGEMM<rowAlg::MaskIndexed_v2> CSR", HeapSpGEMM<rowAlg::MaskIndexed_v2>, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        run("HeapSpGEMM<rowAlg::MaskIndexed_v3> CSR", HeapSpGEMM<rowAlg::MaskIndexed_v3>, innerIters, tnums, flop, A_csr, B_csr, M_csr);
        // @formatter:on
    }

    A_csc.make_empty();
    B_csc.make_empty();
    M_csc.make_empty();
    A_csr.make_empty();
    B_csr.make_empty();
    M_csc.make_empty();
    A_csr_t.make_empty();
    B_csr_t.make_empty();
    M_csr_t.make_empty();

    return 0;
}

