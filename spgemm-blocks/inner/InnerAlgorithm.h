#ifndef MASKED_SPGEMM_INNER_ALGORITHM_H
#define MASKED_SPGEMM_INNER_ALGORITHM_H

template<class IT, class NT, bool Complemented = false>
class MaskedInner {
    static_assert(Complemented == false);

public:
    [[gnu::always_inline]]
    void symbolicRow(const CSR<IT, NT> &A, const CSC<IT, NT> &B, const CSR<IT, NT> &M, IT row, IT *rowNvals) {
        IT currRowNvals = 0;

        for (IT j = M.rowptr[row]; j < M.rowptr[row + 1]; ++j) {
            IT itAIdx = A.rowptr[row];
            const IT rowAEnd = A.rowptr[row + 1];
            if (itAIdx == rowAEnd) { continue; }

            IT itBIdx = B.colptr[M.colids[j]];
            const IT colBEnd = B.colptr[M.colids[j] + 1];
            if (itBIdx == colBEnd) { continue; }

            while (true) {
                if (A.colids[itAIdx] < B.rowids[itBIdx]) {
                    if (++itAIdx == rowAEnd) { break; }
                } else if (A.colids[itAIdx] > B.rowids[itBIdx]) {
                    if (++itBIdx == colBEnd) { break; }
                } else {
                    currRowNvals++;
                    break;
                }
            }
        }

        rowNvals[row] = currRowNvals;
    }

    template<typename MultiplyOperation, typename AddOperation>
    [[gnu::always_inline]]
    void numericRow(const CSR<IT, NT> &A, const CSC<IT, NT> &B, const CSR<IT, NT> &M,
                    MultiplyOperation multop, AddOperation addop, IT row, IT *&currColId, NT *&currValue) {
        for (IT j = M.rowptr[row]; j < M.rowptr[row + 1]; ++j) {
            IT itAIdx = A.rowptr[row];
            const IT rowAEnd = A.rowptr[row + 1];
            if (itAIdx == rowAEnd) { continue; }

            IT itBIdx = B.colptr[M.colids[j]];
            const IT colBEnd = B.colptr[M.colids[j] + 1];
            if (itBIdx == colBEnd) { continue; }

            bool active = false;
            NT value;

            while (true) {
                if (A.colids[itAIdx] < B.rowids[itBIdx]) {
                    if (++itAIdx == rowAEnd) { break; }
                } else if (A.colids[itAIdx] > B.rowids[itBIdx]) {
                    if (++itBIdx == colBEnd) { break; }
                } else {
                    if (active) {
                        value = addop(value, multop(A.values[itAIdx], B.values[itBIdx]));
                    } else {
                        active = true;
                        value = multop(A.values[itAIdx], B.values[itBIdx]);
                    }
                    if (++itAIdx == rowAEnd) { break; }
                    if (++itBIdx == colBEnd) { break; }
                }
            }

            if (active) {
                *(currColId++) = M.colids[j];
                *(currValue++) = value;
            }
        }
    }
};

#endif //MASKED_SPGEMM_INNER_ALGORITHM_H
