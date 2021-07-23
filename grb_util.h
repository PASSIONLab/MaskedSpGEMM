#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>

#include "IO.h"


extern "C"
{
#include "GraphBLAS.h"
}



////////////////////////// Algebraic objects and types /////////////////////////
// C++ does not support _Generic
// #define GrbAlgObj(obj,type) obj##type

template <typename T>
struct GrbAlgObj;



template <>
struct
GrbAlgObj <uint64_t>
{
	auto
	get_type ()
	{
		return GrB_UINT64;
	}



	auto
	get_sr_plus_times ()
	{
		return GrB_PLUS_TIMES_SEMIRING_UINT64;
	}



	auto
	get_sr_plus_land ()
	{
		return GxB_PLUS_LAND_UINT64;
	}



	auto
	get_sr_plus_pair ()
	{
		return GxB_PLUS_PAIR_UINT64;
	}



	auto
	get_sr_plus_second ()
	{
		return GxB_PLUS_SECOND_UINT64;
	}



	auto
	get_unary_set_one ()
	{
		return GxB_ONE_UINT64;
	}



	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_UINT64;
	}



	auto
	get_binary_plus ()
	{
		return GrB_PLUS_UINT64;
	}
};



template <>
struct
GrbAlgObj <int64_t>
{
	auto
	get_type ()
	{
		return GrB_INT64;
	}



	auto
	get_sr_plus_times ()
	{
		return GrB_PLUS_TIMES_SEMIRING_INT64;
	}



	auto
	get_sr_plus_land ()
	{
		return GxB_PLUS_LAND_INT64;
	}



	auto
	get_sr_plus_pair ()
	{
		return GxB_PLUS_PAIR_INT64;
	}



	auto
	get_sr_plus_second ()
	{
		return GxB_PLUS_SECOND_INT64;
	}



	auto
	get_unary_set_one ()
	{
		return GxB_ONE_INT64;
	}



	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_INT64;
	}



	auto
	get_binary_plus ()
	{
		return GrB_PLUS_INT64;
	}
};



template <>
struct
GrbAlgObj <uint32_t>
{
	auto
	get_type ()
	{
		return GrB_UINT32;
	}
	


	auto
	get_sr_plus_times ()
	{
		return GrB_PLUS_TIMES_SEMIRING_UINT32;
	}



	auto
	get_sr_plus_land ()
	{
		return GxB_PLUS_LAND_UINT32;
	}



	auto
	get_sr_plus_second ()
	{
		return GxB_PLUS_SECOND_UINT32;
	}
	


	auto
	get_unary_set_one ()
	{
		return GxB_ONE_UINT32;
	}



	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_UINT32;
	}



	auto
	get_binary_plus ()
	{
		return GrB_PLUS_UINT32;
	}
};



template <>
struct
GrbAlgObj <int32_t>
{
	auto
	get_type ()
	{
		return GrB_INT32;
	}



	auto
	get_sr_plus_times ()
	{
		return GrB_PLUS_TIMES_SEMIRING_INT32;
	}



	auto
	get_sr_plus_land ()
	{
		return GxB_PLUS_LAND_INT32;
	}



	auto
	get_sr_plus_second ()
	{
		return GxB_PLUS_SECOND_INT32;
	}



	auto
	get_unary_set_one ()
	{
		return GxB_ONE_INT32;
	}


	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_INT32;
	}



	auto
	get_binary_plus ()
	{
		return GrB_PLUS_INT32;
	}
};



template <>
struct
GrbAlgObj <float>
{
	auto
	get_type ()
	{
		return GrB_FP32;
	}



	auto
	get_sr_plus_times ()
	{
		return GrB_PLUS_TIMES_SEMIRING_FP32;
	}



	auto
	get_sr_plus_second ()
	{
		return GxB_PLUS_SECOND_FP32;
	}



	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_FP32;
	}



	auto
	get_binary_plus ()
	{
		return GrB_PLUS_FP32;
	}



	auto
	get_binary_times ()
	{
		return GrB_TIMES_FP32;
	}
};



template <>
struct
GrbAlgObj <double>
{
	auto
	get_type ()
	{
		return GrB_FP64;
	}



	auto
	get_sr_plus_times ()
	{
		return GrB_PLUS_TIMES_SEMIRING_FP64;
	}



	auto
	get_sr_plus_second ()
	{
		return GxB_PLUS_SECOND_FP64;
	}




	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_FP64;
	}



	auto
	get_binary_plus ()
	{
		return GrB_PLUS_FP64;
	}



	auto
	get_binary_times ()
	{
		return GrB_TIMES_FP64;
	}
};





//////////////////////////////// matrix builder ////////////////////////////////
// C++ does not support _Generic
template <typename T>
struct GrbMatrixBuild;



template <>
struct
GrbMatrixBuild <uint64_t>
{
	void
	operator() (GrB_Matrix		*A,
				const GrB_Index *rinds,
				const GrB_Index *cinds,
				const uint64_t	*vals,
				GrB_Index		 m,
				GrB_Index		 n,
				GrB_Index		 nnz
				)
	{
		GrB_Matrix_new(A, GrB_UINT64, m, n);
		GrB_Matrix_build_UINT64(*A, rinds, cinds, vals, nnz, GrB_MIN_UINT64);
	}
};



template <>
struct
GrbMatrixBuild <int64_t>
{
	void
	operator() (GrB_Matrix		*A,
				const GrB_Index *rinds,
				const GrB_Index *cinds,
				const int64_t	*vals,
				GrB_Index		 m,
				GrB_Index		 n,
				GrB_Index		 nnz
				)
	{
		GrB_Matrix_new(A, GrB_INT64, m, n);
		GrB_Matrix_build_INT64(*A, rinds, cinds, vals, nnz, GrB_MIN_INT64);
	}
};



template <>
struct
GrbMatrixBuild <uint32_t>
{
	void
	operator() (GrB_Matrix		*A,
				const GrB_Index *rinds,
				const GrB_Index *cinds,
				const uint32_t	*vals,
				GrB_Index		 m,
				GrB_Index		 n,
				GrB_Index		 nnz
				)
	{
		GrB_Matrix_new(A, GrB_UINT32, m, n);
		GrB_Matrix_build_UINT32(*A, rinds, cinds, vals, nnz, GrB_MIN_UINT32);
	}
};



template <>
struct
GrbMatrixBuild <int32_t>
{
	void
	operator() (GrB_Matrix		*A,
				const GrB_Index *rinds,
				const GrB_Index *cinds,
				const int32_t	*vals,
				GrB_Index		 m,
				GrB_Index		 n,
				GrB_Index		 nnz
				)
	{
		GrB_Matrix_new(A, GrB_INT32, m, n);
		GrB_Matrix_build_INT32(*A, rinds, cinds, vals, nnz, GrB_MIN_INT32);
	}
};



template <>
struct
GrbMatrixBuild <float>
{
	void
	operator() (GrB_Matrix		*A,
				const GrB_Index *rinds,
				const GrB_Index *cinds,
				const float 	*vals,
				GrB_Index		 m,
				GrB_Index		 n,
				GrB_Index		 nnz
				)
	{
		GrB_Matrix_new(A, GrB_FP32, m, n);
		GrB_Matrix_build_FP32(*A, rinds, cinds, vals, nnz, GrB_MIN_FP32);
	}
};



template <>
struct
GrbMatrixBuild <double>
{
	void
	operator() (GrB_Matrix		*A,
				const GrB_Index *rinds,
				const GrB_Index *cinds,
				const double 	*vals,
				GrB_Index		 m,
				GrB_Index		 n,
				GrB_Index		 nnz
				)
	{
		GrB_Matrix_new(A, GrB_FP64, m, n);
		GrB_Matrix_build_FP64(*A, rinds, cinds, vals, nnz, GrB_MIN_FP64);
	}
};



//////////////////////////// matrix tuple extractor ////////////////////////////
// C++ does not support _Generic
template <typename T>
struct GrbMatrixExtractTuples;



template <>
struct
GrbMatrixExtractTuples <uint64_t>
{
	void
	operator() (GrB_Index			*rids,
				GrB_Index			*cids,
				uint64_t			*vals,
				GrB_Index			*n,
				const GrB_Matrix	&A
				)
	{
		GrB_Matrix_extractTuples_UINT64(rids, cids, vals, n, A);
	}
};



template <>
struct
GrbMatrixExtractTuples <int64_t>
{
	void
	operator() (GrB_Index			*rids,
				GrB_Index			*cids,
				int64_t  			*vals,
				GrB_Index			*n,
				const GrB_Matrix	&A
				)
	{
		GrB_Matrix_extractTuples_INT64(rids, cids, vals, n, A);
	}
};



template <>
struct
GrbMatrixExtractTuples <uint32_t>
{
	void
	operator() (GrB_Index			*rids,
				GrB_Index			*cids,
				uint32_t			*vals,
				GrB_Index			*n,
				const GrB_Matrix	&A
				)
	{
		GrB_Matrix_extractTuples_UINT32(rids, cids, vals, n, A);
	}
};



template <>
struct
GrbMatrixExtractTuples <int32_t>
{
	void
	operator() (GrB_Index			*rids,
				GrB_Index			*cids,
				int32_t 			*vals,
				GrB_Index			*n,
				const GrB_Matrix	&A
				)
	{
		GrB_Matrix_extractTuples_INT32(rids, cids, vals, n, A);
	}
};



template <>
struct
GrbMatrixExtractTuples <float>
{
	void
	operator() (GrB_Index			*rids,
				GrB_Index			*cids,
				float   			*vals,
				GrB_Index			*n,
				const GrB_Matrix	&A
				)
	{
		GrB_Matrix_extractTuples_FP32(rids, cids, vals, n, A);
	}
};



template <>
struct
GrbMatrixExtractTuples <double>
{
	void
	operator() (GrB_Index			*rids,
				GrB_Index			*cids,
				double  			*vals,
				GrB_Index			*n,
				const GrB_Matrix	&A
				)
	{
		GrB_Matrix_extractTuples_FP64(rids, cids, vals, n, A);
	}
};



/////////////////////////////// matrix reduction ///////////////////////////////
// C++ does not support _Generic
template <typename T>
struct GrbMatrixReduce;



template <>
struct
GrbMatrixReduce <uint64_t>
{
	void
	operator() (uint64_t				*c,
				const GrB_Monoid		 monoid,
				const GrB_Matrix		 A,
				const GrB_Descriptor	 desc
				)
	{
		GrB_Matrix_reduce_UINT64(c, NULL, monoid, A, desc);
	}
};




/////////////////////////////// matrix reduction ///////////////////////////////
// C++ does not support _Generic
template <typename T>
struct GrbMatrixSetElement;


template <>
struct
GrbMatrixSetElement <uint64_t>
{
	void
	operator() (GrB_Matrix	A,
				uint64_t	x,
				GrB_Index	i,
				GrB_Index	j
				)
	{
		GrB_Matrix_setElement_UINT64(A, x, i, j);
	}
};




template <>
struct
GrbMatrixSetElement <int64_t>
{
	void
	operator() (GrB_Matrix	A,
				int64_t		x,
				GrB_Index	i,
				GrB_Index	j
				)
	{
		GrB_Matrix_setElement_INT64(A, x, i, j);
	}
};



template <>
struct
GrbMatrixSetElement <float>
{
	void
	operator() (GrB_Matrix	A,
				float		x,
				GrB_Index	i,
				GrB_Index	j
				)
	{
		GrB_Matrix_setElement_FP32(A, x, i, j);
	}
};



template <>
struct
GrbMatrixSetElement <double>
{
	void
	operator() (GrB_Matrix	A,
				double		x,
				GrB_Index	i,
				GrB_Index	j
				)
	{
		GrB_Matrix_setElement_FP64(A, x, i, j);
	}
};



template <typename NT>
void
read_grb_mtx
(
	GrB_Matrix	*A,
	char		*fpath,
	bool		 remove_diags = false,
	bool		 symmetricize = false,
	bool		 rand_vals	  = false
)
{
	GrB_Index	 m, n, nnz;
	GrB_Index	*rids;
	GrB_Index	*cids;
	NT			*vals;
	ReadASCII_Triples(fpath, m, n, nnz, rids, cids, vals,
					  remove_diags, rand_vals);
	GrbMatrixBuild<NT>()(A, rids, cids, vals, m, n, nnz);
	std::cout << m << " " << n << " " << nnz << std::endl;

	if (symmetricize)
	{
		std::cout << "symmetricizing the matrix" << std::endl;
		GrB_Descriptor desc = NULL;
		GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
		// GrB_Matrix tmp = NULL;
		// GrB_Matrix_new(&tmp, GrbAlgObj<NT>().get_type(), m, n);
		// GrB_transpose(tmp, NULL, NULL, *A, NULL);
		GrB_Matrix_eWiseAdd_Semiring(*A, NULL, NULL,
									 GrbAlgObj<NT>().get_sr_plus_times(),
		 							 *A, *A, desc);
		
		// GxB_Matrix_fprint(tmp, "tmp", GxB_COMPLETE, stdout);
	}

	return;
}

template <typename NT>
void
sort_by_degree
        (
                GrB_Matrix *A
        )
{
    GrB_Index nnz;
    GrB_Index nrows;
    GrB_Index ncols;
    GrB_Matrix_nvals(&nnz, *A);
    GrB_Matrix_nrows(&nrows, *A);
    GrB_Matrix_ncols(&ncols, *A);

    auto rids = new GrB_Index[nnz];
    auto cids = new GrB_Index[nnz];
    auto vals = new NT[nnz];
    GrbMatrixExtractTuples<NT>{}(rids, cids, vals, &nnz, *A);

    auto cnts = new std::pair<GrB_Index, GrB_Index>[nrows];
    for (GrB_Index i = 0; i < nrows; ++i) {
        cnts[i].first = 0;
        cnts[i].second = i;
    }

    for (GrB_Index i = 0; i < nnz; ++i) {
        ++cnts[rids[i]].first;
    }
    std::sort(cnts, cnts + nrows, std::greater<>{});

    auto mapping = new GrB_Index[nrows];
    for (GrB_Index i = 0; i < nrows; i++) {
        mapping[cnts[i].second] = i;
    }

    for (GrB_Index i = 0; i < nnz; ++i) {
        rids[i] = mapping[rids[i]];
        cids[i] = mapping[cids[i]];
    }

    delete[] mapping;
    delete[] cnts;

    GrbMatrixBuild<NT>()(A, rids, cids, vals, nrows, ncols, nnz);

    delete[] rids;
    delete[] cids;
    delete[] vals;
}

template <typename NT>
void
get_lowtri
(
    GrB_Matrix A,
    GrB_Matrix *L
)
{
	GrbAlgObj<NT> to_grb;
	GxB_Scalar Thunk = NULL;
	GrB_Index n;
	GrB_Matrix_nrows(&n, A);	

	GxB_Scalar_new(&Thunk, GrB_INT64);
	GrB_Matrix_new(L, to_grb.get_type(), n, n);
	GxB_Scalar_setElement_INT64(Thunk, (int64_t) (-1));
	GxB_Matrix_select(*L, NULL, NULL, GxB_TRIL, A, Thunk, NULL);

	return;	
}

template <typename NT>
void
convert_to_csc
(
        GrB_Matrix A,
        GrB_Matrix *A_csc
) {
    GrB_Index nnz;
    GrB_Index nrows;
    GrB_Index ncols;
    GrB_Matrix_nvals(&nnz, A);
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);

    auto rids = new GrB_Index[nnz];
    auto cids = new GrB_Index[nnz];
    auto vals = new NT[nnz];
    GrbMatrixExtractTuples<NT>{}(rids, cids, vals, &nnz, A);

    GxB_Format_Value old;
    GxB_Global_Option_get(GxB_FORMAT, &old);
    GxB_Global_Option_set(GxB_FORMAT, GxB_BY_COL);

    GrB_Matrix_new(A_csc, GrbAlgObj<NT>{}.get_type(), nrows, ncols);
    GrbMatrixBuild<NT>{}(A_csc, rids, cids, vals, nrows, ncols, nnz);

    GxB_Global_Option_set(GxB_FORMAT, old);

    delete[] rids;
    delete[] cids;
    delete[] vals;
}



template <typename NT>
void
avoid_iso
(
    GrB_Matrix	A,
 	NT			x,
 	GrB_Index	i,
 	GrB_Index	j
)
{
	GrbMatrixSetElement<NT>()(A, static_cast<NT>(0), i, j);
	GrbMatrixSetElement<NT>()(A, x, i, j);
}
