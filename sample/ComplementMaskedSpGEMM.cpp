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
#include "sample-util.h"
#include "../spa_mult.h"
#include "../spgemm-blocks/masked-spgemm.h"

using namespace std;

#define RUN_CSR_IMPL(NAME, FUNC) run(fileName, NAME, FUNC, warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr)
#define RUN_CSR(ALG) RUN_CSR_IMPL(#ALG, ALG)
#define RUN_CSR_1P(ALG) RUN_CSR_IMPL(#ALG "-1P", MaskedSpGEMM1p<ALG>)
#define RUN_CSR_2P(ALG) RUN_CSR_IMPL(#ALG "-2P", MaskedSpGEMM2p<ALG>)

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

    string fileName = argv[1];
    CSC<Index_t, Value_t> A_csc;
    ReadASCII(fileName, A_csc);
    CSR<Index_t, Value_t> A_csr(A_csc); //converts, allocates and populates

    // @formatter:off
    size_t outerIters  = std::getenv("OUTER_ITERS")  ? std::stoul(std::getenv("OUTER_ITERS"))  : 1;
    size_t innerIters  = std::getenv("INNER_ITERS")  ? std::stoul(std::getenv("INNER_ITERS"))  : 1;
    size_t warmupIters = std::getenv("WARMUP_ITERS") ? std::stoul(std::getenv("WARMUP_ITERS")) : (innerIters == 1 ? 0 : 1);
    string mode        = std::getenv("MODE")         ? std::getenv("MODE")                     : "";
    // @formatter:on

    if (mode.empty()) { std::cerr << "Mode unspecified!" << std::endl; }
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return std::tolower(c); });

    std::cout << "Iters: " << outerIters << " x (" << warmupIters << "," << innerIters << ")" << std::endl << std::endl;

    std::size_t flop = get_flop(A_csc, A_csc);
    for (size_t i = 0; i < outerIters; i++) {
        if (mode == "heap" || mode == "all") {
            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<true, true, 0>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<true, true, 0>::Impl>));
        }

        if (mode == "hash" || mode == "all") {
            RUN_CSR((MaskedSpGEMM1p<MaskedHash<true, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHash<true, false>::Impl>));
        }

        if (mode == "msa" || mode == "all") {
            RUN_CSR((MaskedSpGEMM1p<MSA1A<true, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MSA1A<true, false>::Impl>));

            RUN_CSR((MaskedSpGEMM1p<MSA2A<true, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MSA2A<true, false>::Impl>));
        }
    }

    A_csc.make_empty();
    A_csr.make_empty();

    return 0;
}
