#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>



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
	get_unary_set_one ()
	{
		return GxB_ONE_UINT64;
	}



	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_UINT64;
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
	get_unary_set_one ()
	{
		return GxB_ONE_INT64;
	}



	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_INT64;
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
	get_unary_set_one ()
	{
		return GxB_ONE_UINT32;
	}



	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_UINT32;
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
	get_unary_set_one ()
	{
		return GxB_ONE_INT32;
	}


	auto
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_INT32;
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
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_FP32;
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
	get_monoid_plus ()
	{
		return GrB_PLUS_MONOID_FP64;
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




template <typename NT>
void
read_grb_mtx
(
	GrB_Matrix *A,
	char       *fpath,
	bool		remove_diags = false,
	bool		symmetricize = false
)
{
	GrB_Index	 m, n, nnz;
	GrB_Index	*rids;
	GrB_Index	*cids;
	NT			*vals;
	ReadASCII_Triples(fpath, m, n, nnz, rids, cids, vals, remove_diags);
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
	GrB_Matrix_apply(*L, NULL, NULL, to_grb.get_unary_set_one(), *L, NULL);


	return;	
}
