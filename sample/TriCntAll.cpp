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
#include "../spgemm-blocks/inner/masked-spgemm-inner.h"
#include "../spgemm-blocks/inner/MaskedInnerSpGEMM.h"

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
    uint16_t ptrCSC;
    if constexpr (std::is_same<AT<IT, NT>, CSR<IT, NT>>::value) {
        idsCSC = calculateChecksum(reinterpret_cast<uint8_t *>(A.colids), A.nnz * sizeof(IT));
        ptrCSC = calculateChecksum(reinterpret_cast<uint8_t *>(A.rowptr), A.rows * sizeof(IT));
    } else {
        idsCSC = calculateChecksum(reinterpret_cast<uint8_t *>(A.rowids), A.nnz * sizeof(IT));
        ptrCSC = calculateChecksum(reinterpret_cast<uint8_t *>(A.colptr), A.cols * sizeof(IT));
    }

    return to_string(ptrCSC) + "|" + to_string(valuesCSC) + "|" + to_string(idsCSC);
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
         size_t witers, size_t niters, vector<int> &tnums, size_t nfop,
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
        double mflops = (double) nfop / ave_msec / 1000;

        std::cout << "LOG," << fileName << "," << name << "," << typeid(IT).name() << "|" << typeid(NT).name()
                  << "," << tnum << "," << ave_msec << "," << mflops << ","
                  << C.nnz << "," << C.sumall() << "," << checksum(C) << std::endl;

        C.make_empty();
    }
}

template<class IT, class NT>
void setRowData(const CSR<IT, NT> &A, const CSR<IT, NT> &B, const CSR<IT, NT> &M,
                std::vector<long *> &data) {
    using AccumulatorT = MaskedSparseAccumulator2A<IT, void>;

    auto ncols = new long[A.rows];
    auto rowSizesA = new long[A.rows];
    auto rowSizesM = new long[A.rows];
    auto rowFlops = new long[A.rows];
    auto rowFlopsMasked = new long[A.rows];
    auto rowNvals = new long[A.rows];

    data.push_back(ncols);
    data.push_back(rowSizesA);
    data.push_back(rowSizesM);
    data.push_back(rowFlops);
    data.push_back(rowFlopsMasked);
    data.push_back(rowNvals);

#pragma omp parallel default(none) shared(A, B, M, ncols, rowSizesA, rowSizesM, rowFlops, rowFlopsMasked, rowNvals)
    {
        // Initialize SPA
        AccumulatorT spa(A.cols);
        auto[spaSize, spaAlignment] = spa.getMemoryRequirement();
        auto mem = mallocAligned(spaSize);
        spa.setBuffer(mem, spaSize, spaSize);

#pragma omp parallel for default(none) shared(A, B, M, ncols, rowSizesA, rowSizesM, rowFlops, rowFlopsMasked, rowNvals, spa)
        for (size_t row = 0; row < A.rows; ++row) {
            ncols[row] = B.cols;
            rowSizesA[row] = A.rowptr[row + 1] - A.rowptr[row];
            rowSizesM[row] = M.rowptr[row + 1] - M.rowptr[row];

            // Calculate flops
            IT currRowFlops = 0;
            for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; ++j) {
                IT inner = A.colids[j];
                IT npins = B.rowptr[inner + 1] - B.rowptr[inner];
                currRowFlops += npins;
            }
            rowFlops[row] = currRowFlops;

            // Calculate nvals and nflopsMasked
            {
                const auto maskBegin = &M.colids[M.rowptr[row]];
                const auto maskEnd = &M.colids[M.rowptr[row + 1]];

                // Insert all mask elements to the SPA
                for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
                    spa.setAllowed(*maskIt);
                }

                IT currRowNvals = 0;
                IT currRowNflops = 0;
                for (IT j = A.rowptr[row]; j < A.rowptr[row + 1]; j++) {
                    IT inner = A.colids[j];
                    for (IT k = B.rowptr[inner]; k < B.rowptr[inner + 1]; k++) {
                        auto &state = spa.getState(B.colids[k]);
                        if (state == AccumulatorT::EMPTY) { continue; }

                        if (state == AccumulatorT::ALLOWED) {
                            currRowNvals++;
                            currRowNflops++;
                            state = AccumulatorT::INITIALIZED;
                        } else {
                            assert(state == AccumulatorT::INITIALIZED);
                            currRowNflops++;
                        }
                    }
                }
                // Reset - Remove all mask elements from the SPA

                for (auto maskIt = maskBegin; maskIt != maskEnd; maskIt++) {
                    spa.clear(*maskIt);
                }
                rowFlopsMasked[row] = currRowNflops;
                rowNvals[row] = currRowNflops;
            }
        }

        freeAligned(mem);
    }


}

void printRowData(std::vector<long *> &data, size_t nrows, size_t niter, size_t nsamples,
                  const std::vector<std::string> &algorithmNames) {
    size_t nh = data.size() - niter * algorithmNames.size();

    std::cout << "CSV" << std::endl;

    std::cout << "NCOLS, ROW_SIZE_A, ROW_SIZE_M, ROW_FLOPS, ROW_FLOPS_MASKED, ROW_NVALS";
    for (const auto &alg : algorithmNames) { std::cout << ", " << alg; }
    std::cout << std::endl;

    std::vector<long> times(niter);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < nh; j++) {
            if (j) { std::cout << ", "; }
            std::cout << data[j][i];
        }

        for (size_t algNum = 0; algNum < algorithmNames.size(); algNum++) {
            for (size_t iter = 0; iter < niter; iter++) {
                times[iter] = data[nh + algNum * niter + iter][i];
            }
            std::sort(times.begin(), times.end());
            long avg = std::accumulate(times.begin(), times.begin() + nsamples, 0) / nsamples;
            std::cout << ", " << avg;
        }

        std::cout << std::endl;
    }
}

template<class IT, class NT,
        template<class, class> class AT,
        template<class, class> class BT,
        template<class, class> class CT = AT,
        template<class, class> class MT>
void profile(const std::string &name,
             void(*f)(long *, const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &, const MT<IT, NT> &,
                      multiplies<NT>, plus<NT>, unsigned),
             size_t witers, size_t niters, vector<int> &tnums, size_t nfop,
             const AT<IT, NT> &A, const BT<IT, NT> &B, const MT<IT, NT> &M, std::vector<long *> &data) {
    for (int tnum : tnums) {
        omp_set_num_threads(tnum); // TODO: update get_flop to use numThreads methods and remove this

        CT<IT, NT> C;

        // The first iteration is excluded from evaluation if there is only one iteration
        for (int i = 0; i < witers; ++i) {
            auto times = new long[A.rows];
            f(times, A, B, C, M, multiplies<NT>(), plus<NT>(), tnum);
            delete[] times;
        }

        double ave_msec = 0;
        for (int i = 0; i < niters; ++i) {
            C.make_empty();

            auto times = new long[A.rows];
            double start = omp_get_wtime();
            f(times, A, B, C, M, multiplies<NT>(), plus<NT>(), tnum);
            double end = omp_get_wtime();
            data.push_back(times);

            double msec = (end - start) * 1000;
            ave_msec += msec;
        }

        ave_msec /= double(niters);
        double mflops = (double) nfop / ave_msec / 1000;

        std::cout << "LOG-prof," << fileName << "," << name << "," << typeid(IT).name() << "|" << typeid(NT).name()
                  << "," << tnum << "," << ave_msec << "," << mflops << ","
                  << C.nnz << "," << C.sumall() << "," << checksum(C) << std::endl;

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

#define RUN_CSR_IMPL(NAME, FUNC) run(NAME, FUNC, warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr)
#define RUN_CSR(ALG) RUN_CSR_IMPL(#ALG, ALG)
#define RUN_CSR_1P(ALG) RUN_CSR_IMPL(#ALG "-1P", MaskedSpGEMM1p<ALG>)
#define RUN_CSR_2P(ALG) RUN_CSR_IMPL(#ALG "-2P", MaskedSpGEMM2p<ALG>)

#define RUN_CSR_CSC_IMPL(NAME, FUNC) run(NAME, FUNC, warmupIters, innerIters, tnums, flop, A_csr, A_csc, A_csr)
#define RUN_CSR_CSC(ALG) RUN_CSR_CSC_IMPL(#ALG, ALG)

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

    // @formatter:off
    size_t outerIters  = std::getenv("OUTER_ITERS")  ? std::stoul(std::getenv("OUTER_ITERS"))  : 1;
    size_t innerIters  = std::getenv("INNER_ITERS")  ? std::stoul(std::getenv("INNER_ITERS"))  : 1;
    size_t warmupIters = std::getenv("WARMUP_ITERS") ? std::stoul(std::getenv("WARMUP_ITERS")) : (innerIters == 1 ? 0 : 1);
    string mode        = std::getenv("MODE")         ? std::getenv("MODE")                         : "";
    // @formatter:on

    std::cout << "Iters: " << outerIters << " x (" << warmupIters << "," << innerIters << ")" << std::endl << std::endl;

    std::size_t flop = get_flop(A_csc, A_csc);

    for (size_t i = 0; i < outerIters; i++) {
        if (mode == "Inner" || mode == "Dot") {
            RUN_CSR_CSC((innerSpGEMM_nohash<false, false>));
            RUN_CSR_CSC(MaskedSpGEMM1p);
            RUN_CSR_CSC(MaskedSpGEMM2p);
            RUN_CSR_CSC(MaskedSpGEMM1p<MaskedInner>);
            RUN_CSR_CSC(MaskedSpGEMM2p<MaskedInner>);
        } else if (mode == "SPA") {
            RUN_CSR(MaskedSPASpGEMM);
            RUN_CSR_1P(MSA2A);
            RUN_CSR_2P(MSA2A);
            RUN_CSR_1P(MSA1A);
            RUN_CSR_2P(MSA1A);
        } else if (mode == "Hash") {
            RUN_CSR(mxm_hash_mask_wobin);
            RUN_CSR(mxm_hash_mask);
            RUN_CSR_1P(MaskedHash);
            RUN_CSR_2P(MaskedHash);
        } else if (mode == "Heap") {
            RUN_CSR_1P(MaskedHeap_v0);
            RUN_CSR_2P(MaskedHeap_v0);
            RUN_CSR_1P(MaskedHeap<0>::Impl);
            RUN_CSR_1P(MaskedHeap_v1);
            RUN_CSR_2P(MaskedHeap_v1);
            RUN_CSR_1P(MaskedHeap<1>::Impl);
            RUN_CSR_1P(MaskedHeap<2>::Impl);
            RUN_CSR_1P(MaskedHeap<8>::Impl);
            RUN_CSR_1P(MaskedHeap<32>::Impl);
            RUN_CSR_1P(MaskedHeap<128>::Impl);
            RUN_CSR_1P(MaskedHeap<512>::Impl);
            RUN_CSR_1P(MaskedHeap_v2);
            RUN_CSR_2P(MaskedHeap_v2);
            RUN_CSR_1P(MaskedHeap<std::numeric_limits<std::size_t>::max()>::Impl);
        } else if (mode == "All1p") {
            RUN_CSR_CSC((innerSpGEMM_nohash<false, false>));
            RUN_CSR_1P(MaskedHeap_v1);
            RUN_CSR_1P(MaskedHeap_v2);
            RUN_CSR_1P(MaskedHash);
            RUN_CSR_1P(MSA2A);
            RUN_CSR_1P(MCA);
            RUN_CSR(MaskedSpGEMM1p);
        } else if (mode == "Prof") {
            std::vector<long *> data;
            setRowData(A_csr, A_csr, A_csr, data);

            RUN_CSR_1P(MSA2A);
            RUN_CSR_1P(MaskedHash);
            RUN_CSR_1P(MCA);
            RUN_CSR_1P(MaskedHeap_v1);
            RUN_CSR_1P(MaskedHeap_v2);

            // @formatter:off
            profile("MaskedSpGEMM1p<MSA2A>",         MaskedSpGEMM1p_prof<MSA2A>,         warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr, data);
            profile("MaskedSpGEMM1p<MaskedHash>",    MaskedSpGEMM1p_prof<MaskedHash>,    warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr, data);
            profile("MaskedSpGEMM1p<MCA>",           MaskedSpGEMM1p_prof<MCA>,           warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr, data);
            profile("MaskedSpGEMM1p<MaskedHeap_v1>", MaskedSpGEMM1p_prof<MaskedHeap_v1>, warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr, data);
            profile("MaskedSpGEMM1p<MaskedHeap_v2>", MaskedSpGEMM1p_prof<MaskedHeap_v2>, warmupIters, innerIters, tnums, flop, A_csr, A_csr, A_csr, data);
            // @formatter:on

            printRowData(data, A_csr.rows, innerIters, (innerIters + 1) / 2,
                         {"MSA2A", "MaskedHash", "MCA", "MaskedHeap_v1", "MaskedHeap_v2"});
        } else {
            std::cout << "LOG,start" << std::endl;
            RUN_CSR_CSC((innerSpGEMM_nohash<false, false>));
            std::cout << "LOG,separator" << std::endl;
            RUN_CSR(mxm_hash_mask_wobin);
            RUN_CSR(mxm_hash_mask);
            RUN_CSR_1P(MaskedHash);
            RUN_CSR_2P(MaskedHash);
            std::cout << "LOG,separator" << std::endl;
            RUN_CSR(MaskedSPASpGEMM);
            RUN_CSR_1P(MSA2A);
            RUN_CSR_2P(MSA2A);
            RUN_CSR_1P(MSA1A);
            RUN_CSR_2P(MSA1A);
            std::cout << "LOG,separator" << std::endl;
            RUN_CSR(HeapSpGEMM<rowAlg::MCA_v1>);
            RUN_CSR(HeapSpGEMM<rowAlg::MCA_v2>);
            RUN_CSR(HeapSpGEMM<rowAlg::MCA_v3>);
            RUN_CSR_1P(MCA);
            RUN_CSR_2P(MCA);
            std::cout << "LOG,separator" << std::endl;
            RUN_CSR(HeapSpGEMM<rowAlg::MaskedHeap_v0>);
            RUN_CSR(HeapSpGEMM<rowAlg::MaskedHeap_v1>);
            RUN_CSR(HeapSpGEMM<rowAlg::MaskedHeap_v2>);
            RUN_CSR_1P(MaskedHeap_v0);
            RUN_CSR_2P(MaskedHeap_v0);
            RUN_CSR_1P(MaskedHeap_v1);
            RUN_CSR_2P(MaskedHeap_v1);
            RUN_CSR_1P(MaskedHeap_v2);
            RUN_CSR_2P(MaskedHeap_v2);
        }
    }

    A_csc.make_empty();
    A_csr.make_empty();

    return 0;
}
