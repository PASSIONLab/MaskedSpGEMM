#ifndef MASKED_SPGEMM_SAMPLE_UTIL_H
#define MASKED_SPGEMM_SAMPLE_UTIL_H

#include <iostream>
#include <string>

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

bool startsWith(std::string const &str, std::string const &prefix) {
    if (prefix.size() > str.size()) { return false; }
    if (str.compare(0, prefix.length(), prefix) != 0) { return false; }
    return true;
}

inline std::string removePrefix(std::string const &str, std::string const &prefix) {
    if (prefix.size() > str.size()) { return str; }

    if (str.compare(0, prefix.length(), prefix) == 0) {
        return str.substr(prefix.size());
    }
    return str;
}

inline std::string removeSuffix(std::string const &str, std::string const &suffix) {
    if (suffix.size() > str.size()) { return str; }

    if (str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0) {
        return str.substr(0, str.length() - suffix.length());
    }
    return str;
}

std::string processAlgorithmName(std::string name) {
    // Remove parentheses
    if (name.front() == '(' && name.back() == ')') { name = name.substr(1, name.size() - 2); }

    // Remove MaskedSpGEMMxp
    int numPhases = 0;
    if (startsWith(name, "MaskedSpGEMM1p<")) { numPhases = 1; }
    else if (startsWith(name, "MaskedSpGEMM2p<")) { numPhases = 2; }

    name = removePrefix(name, "MaskedSpGEMM1p<");
    name = removePrefix(name, "MaskedSpGEMM2p<");
    name = removeSuffix(name, ">");

    // Remove ::Impl
    const std::string implStr = "::Impl";
    name = removeSuffix(name, "::Impl");

    if (numPhases == 1) { name += "-1P"; } else if (numPhases == 2) { name += "-2P"; }

    return name;
}

std::string getFileName(const std::string &path) {
    auto pos = path.rfind('/');
    return pos != std::string::npos ? path.substr(pos + 1) : path;
}

template<class IT, class NT,
        template<class, class> class AT,
        template<class, class> class BT,
        template<class, class> class CT = AT,
        template<class, class> class MT>
void run(const std::string &inputName, const std::string &algorithmName,
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

        std::cout << "LOG,"
                  << std::setw(20) << getFileName(inputName) << ","
                  << std::setw(50) << processAlgorithmName(algorithmName) << ","
                  << std::setw(5) << (std::string(typeid(IT).name()) + "|" + std::string(typeid(NT).name())) << ","
                  << std::setw(5) << tnum << ","
                  << std::setw(15) << std::setprecision(4) << std::fixed << ave_msec << ","
                  << std::setw(15) << std::setprecision(4) << std::fixed << mflops << ","
                  << std::setw(10) << C.nnz << ","
                  << std::setw(10) << C.sumall() << ","
                  << std::setw(20) << checksum(C) << std::endl;

        C.make_empty();
    }
}

#endif //MASKED_SPGEMM_SAMPLE_UTIL_H
