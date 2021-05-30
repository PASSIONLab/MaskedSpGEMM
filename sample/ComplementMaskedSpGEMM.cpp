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
#include "../spgemm-blocks/masked-spgemm-poly.h"

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
    size_t len = strlen(path);
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
         size_t witers, size_t niters, vector<int> &tnums, size_t nflop,
         const AT<IT, NT> &A, const BT<IT, NT> &B, const MT<IT, NT> &M) {
    for (int tnum : tnums) {
        omp_set_num_threads(tnum); // TODO: update get_flop to use numThreads methods and remove this

        CT<IT, NT> C;

        // The first iteration is excluded from evaluation if there is only one iteration
        for (int i = 0; i < witers; ++i) { f(A, B, C, M, multiplies<NT>(), plus<NT>(), tnum); }

        double ave_msec = 0;
        for (int i = 0; i < niters; ++i) {
            C.make_empty();

            double start = omp_get_wtime();
            f(A, B, C, M, multiplies<NT>(), plus<NT>(), tnum);
            double end = omp_get_wtime();

            double msec = (end - start) * 1000;
            ave_msec += msec;
        }

        ave_msec /= static_cast<double>(niters);
        double mflops = (double) nflop / ave_msec / 1000;

        std::cout << "LOG," << fileName << "," << name << "," << typeid(IT).name() << "|" << typeid(NT).name()
                  << "," << tnum << "," << ave_msec << "," << mflops << ","
                  << C.nnz << "," << C.sumall() << "," << checksum(C) << std::endl;

        C.make_empty();
    }
}

#define RUN_CSR_IMPL(NAME, FUNC) run(NAME, FUNC, warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr)
#define RUN_CSR_1P(ALG, COMPLEMENTED) RUN_CSR_IMPL(#ALG "-1P", (MaskedSpGEMM1p<ALG, COMPLEMENTED>))
#define RUN_CSR_2P(ALG, COMPLEMENTED) RUN_CSR_IMPL(#ALG "-2P", (MaskedSpGEMM2p<ALG, COMPLEMENTED>))

int main(int argc, char *argv[]) {
    using Value_t = long unsigned;
    using Index_t = long;

    vector<int> tnums;
    if (argc < 3) {
        cout << "Normal usage: ./" << getFileName(argv[0]) << " matrix1.mtx <numthreads>" << endl;
        return -1;
    } else {
        cout << "Running on " << argv[2] << " processors" << endl << endl;
        tnums = {atoi(argv[2])};
    }

    fileName = getFileName(argv[1]);
    string inputname1 = argv[1];
    CSC<Index_t, Value_t> A_csc;
    ReadASCII(inputname1, A_csc);
    CSR<Index_t, Value_t> A_csr(A_csc); //converts, allocates and populates

    // @formatter:off
    size_t outerIters  = std::getenv("OUTER_ITERS")  ? std::stoul(std::getenv("OUTER_ITERS"))  : 1;
    size_t innerIters  = std::getenv("INNER_ITERS")  ? std::stoul(std::getenv("INNER_ITERS"))  : 1;
    size_t warmupIters = std::getenv("WARMUP_ITERS") ? std::stoul(std::getenv("WARMUP_ITERS")) : (innerIters == 1 ? 0 : 1);
    string mode        = std::getenv("MODE")         ? std::getenv("MODE")                         : "";
    // @formatter:on

    std::cout << "Iters: " << outerIters << " x (" << warmupIters << "," << innerIters << ")" << std::endl << std::endl;

    std::size_t flop = get_flop(A_csc, A_csc);

    for (size_t i = 0; i < outerIters; i++) {
        if (mode == "Heap") {
            RUN_CSR_1P(MaskedHeap_v0, true);
        } else {
            std::cerr << "Mode unspecified!" << std::endl;
        }
    }

    A_csc.make_empty();
    A_csr.make_empty();

    return 0;
}
