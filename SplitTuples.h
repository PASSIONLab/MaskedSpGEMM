#ifndef _SplitTuples_H_
#define _SplitTuples_H_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <tuple>
#include <numeric>

#include "CSC.h"
#include "CSR.h"

using namespace std;

template <class IT, class NT>
class SplitTuples
{
public:

    SplitTuples (CSC<IT,NT>& A, int numsplits);
    SplitTuples (CSR<IT,NT>& A, int numsplits);
    
    IT rows;
    IT cols;
    IT nnz;
    int splits;
    std::vector<std::vector<std::tuple<IT, IT, NT>>> splitTuples;
};


template <class IT, class NT>
SplitTuples<IT,NT>::SplitTuples(CSC<IT, NT> & A, int numsplits)
{
    rows = A.rows;
    cols = A.cols;
    nnz = A.nnz;
    splits = numsplits;
    splitTuples.resize(numsplits);

    IT perpiece = A.rows / splits;
    for (IT i=0; i < A.cols; ++i)
    {
        for (IT j = A.colptr[i]; j < A.colptr[i+1]; ++j)
        {
            IT rowid = A.rowids[j];
            IT owner = std::min(rowid / perpiece, static_cast<IT>(numsplits-1));
            splitTuples[owner].push_back(std::make_tuple(rowid, i, A.values[j]));
        }
    }
}

template <class IT, class NT>
SplitTuples<IT,NT>::SplitTuples(CSR<IT, NT> & A, int numsplits)
{
    rows = A.rows;
    cols = A.cols;
    nnz = A.nnz;
    splits = numsplits;
    splitTuples.resize(numsplits);

    IT perpiece = A.cols / splits;
    for (IT i=0; i < A.rows; ++i)
    {
        IT owner = std::min(i / perpiece, static_cast<IT>(numsplits-1));
        for (IT j = A.rowptr[i]; j < A.rowptr[i+1]; ++j)
        {
            IT colid = A.colids[j];
            splitTuples[owner].push_back(std::make_tuple(i % perpiece, colid, A.values[j]));
        }
    }
}
#endif
