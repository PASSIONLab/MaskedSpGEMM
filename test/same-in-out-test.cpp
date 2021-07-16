#include "../CSR.h"
#include "../spgemm-blocks/masked-spgemm.h"

template<class IT, class NT>
void print(const CSR<IT, NT> &A) {
    std::cout << "Matrix:" << std::endl;
    for (int row = 0; row < A.rows; ++row) {
        for (int col = A.rowptr[row]; col < A.rowptr[row + 1]; ++col) {
            std::cout << "\t" << row << " " << A.colids[col] << " " << A.values[col] << std::endl;
        }
    }
    std::cout << std::endl;
}

int main() {
    CSR<int, int> A{"/home/sm108/projects/grb-fusion/apps/inputs/simple"};
    CSR<int, int> C;

    print(A);

    MaskedSpGEMM1p<MSA1A<false, false>::Impl>(A, A, C, A, std::multiplies{}, std::plus{}, 1);
    MaskedSpGEMM1p<MSA1A<false, false>::Impl>(A, A, A, A, std::multiplies{}, std::plus{}, 1);

    print(C);
    print(A);
}