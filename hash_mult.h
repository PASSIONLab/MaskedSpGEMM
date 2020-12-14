/**
 * @file
 *	hash_mult.h
 *
 * @author
 *
 * @date
 *
 * @brief
 *	Masked SpGEMM with hash tables as accumulators
 *
 * @todo
 *
 * @note
 *	
 */

#include <algorithm>

#include <omp.h>

#include "CSR.h"
#include "ht.h"
#include "utility.h"



template <typename IT,
		  typename NT,
		  typename MultiplyOperation,
		  typename AddOperation>
void
mxm_hash_mask
(
	const CSR<IT, NT>	&A,
	const CSR<IT, NT>	&B,
	CSR<IT, NT>			&C,
	CSR<IT, NT>			&M,
	MultiplyOperation	 multop,
	AddOperation		 addop
)
{
	C.rows		   = A.rows;
    C.cols		   = B.cols;
    C.zerobased	   = true;
    C.rowptr	   = my_malloc<IT>(C.rows + 1);
    IT *row_nz     = my_malloc<IT>(C.rows);

	#pragma omp parallel for
	for (IT ra = 0; ra < A.rows; ++ra)
	{
		map_lp<IT, bool> ht(M.rowptr[ra+1] - M.rowptr[ra] + 1);
		for (IT cmptr = M.rowptr[ra]; cmptr < M.rowptr[ra+1]; ++cmptr)
			ht.insert(M.colids[cmptr], false);

		row_nz[ra] = 0;
		for (IT captr = A.rowptr[ra]; captr < A.rowptr[ra+1]; ++captr)
		{
			IT rb = A.colids[captr];
			for (IT cbptr = B.rowptr[rb]; cbptr < B.rowptr[rb+1]; ++cbptr)
			{
				auto hv = ht.find(B.colids[cbptr]);
				if (hv != -1 && !ht[hv])
				{
					++row_nz[ra];
					ht[hv] = true;
				}
			}
		}
	}
	
	scan(row_nz, C.rowptr, C.rows + 1);
    my_free<IT>(row_nz);
	C.nnz	 = C.rowptr[C.rows];    
    C.colids = my_malloc<IT>(C.nnz);
    C.values = my_malloc<NT>(C.nnz);

	#pragma omp parallel for
	for (IT ra = 0; ra < A.rows; ++ra)
	{
		map_lp<IT, NT, bool> ht(M.rowptr[ra+1] - M.rowptr[ra] + 1);
		for (IT cmptr = M.rowptr[ra]; cmptr < M.rowptr[ra+1]; ++cmptr)
			ht.insert(M.colids[cmptr], NT(), false);

		for (IT captr = A.rowptr[ra]; captr < A.rowptr[ra+1]; ++captr)
		{
			IT rb = A.colids[captr];
			for (IT cbptr = B.rowptr[rb]; cbptr < B.rowptr[rb+1]; ++cbptr)
			{
				auto hv = ht.find(B.colids[cbptr]);
				if (hv != -1 && ht.get2(hv))
				{
					ht.get1(hv) = multop(A.values[captr], B.values[cbptr]);
					ht.get2(hv) = true;
				}
				else if (hv != -1 && ht.get2(hv))
					ht.get1(hv) =
						addop(ht.get1(hv),
							  multop(A.values[captr], B.values[cbptr]));
			}
		}

		ht.gather(C.colids + C.rowptr[ra], C.values + C.rowptr[ra]);
	}


	return;
}
