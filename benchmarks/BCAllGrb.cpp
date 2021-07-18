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
#include "../sample/sample_common.hpp"
#include "benchmarks-util.h"
#include "../spa_mult.h"
#include "../spgemm-blocks/masked-spgemm.h"
#include "../spgemm-blocks/masked-spgemm-prof.h"
#include "../spgemm-blocks/masked-spgemm-poly.h"
#include "../spgemm-blocks/masked-spgemm-inner.h"
#include "../spgemm-blocks/inner/MaskedInnerSpGEMM.h"
#include "../multiply.h"

using namespace std;



template <class IT,
		  class NT>
void
grb_bc
(
  	const std::string	&inputName,
	size_t				 witers,
	size_t				 niters,
	vector<int>			&tnums,
	GrB_Vector			*delta,
	GrB_Matrix			 A,
	vector<GrB_Index>	&srcs
)
{
	using				RT = float;
	GrB_Index			n, s, nnz;
	GrbAlgObj<NT>		to_grb;
	GrbAlgObj<RT>		to_grb_real;
	GrB_Matrix			F  = NULL, N = NULL, AT = NULL;
	vector<GrB_Matrix>	preds;

	GrB_Descriptor desc_mxm_01 = NULL, desc_mxm_02 = NULL;
	GrB_Descriptor_new(&desc_mxm_01);
	GxB_Desc_set(desc_mxm_01, GxB_SORT, 1);
	GxB_Desc_set(desc_mxm_01, GrB_OUTP, GrB_REPLACE);
	GxB_Desc_set(desc_mxm_01, GrB_MASK, GrB_COMP);
	GrB_Descriptor_new(&desc_mxm_02);
	GxB_Desc_set(desc_mxm_02, GxB_SORT, 1);
	GxB_Desc_set(desc_mxm_02, GrB_OUTP, GrB_REPLACE);
	// GxB_Desc_set(desc_mxm, GrB_MASK, GrB_STRUCTURE);

	s = srcs.size();
	GrB_Matrix_nrows(&n, A);
	GrB_Vector_new(delta, to_grb_real.get_type(), n);

	vector<GrB_Index> cids(s);
	std::iota(cids.begin(), cids.end(), 0);

	// frontier and shortest path matrices
	vector<NT> vals(s, 1);
	GrbMatrixBuild<NT>()(&F, srcs.data(), cids.data(), vals.data(), n, s, s);
	GrB_Matrix_dup(&N, F);
	GrB_Matrix_new(&AT, to_grb.get_type(), n, n);
	GrB_transpose(AT, NULL, NULL, A, NULL);

	// sigma
	GrB_Index	nvals = s;
	int32_t		depth = 0;
	while (nvals > 0)
	{
		GrB_mxm(F, N, NULL, to_grb.get_sr_plus_second(), AT, F, desc_mxm_01);

		GrB_Matrix_nvals(&nvals, F);

		GrB_Matrix Ftmp = NULL;
		GrB_Matrix_dup(&Ftmp, F);
		preds.push_back(Ftmp);

		GrB_Matrix_eWiseAdd_BinaryOp(N, NULL, NULL, to_grb.get_binary_plus(),
									 N, F, NULL);

		++depth;
	}

	// cout << "##sigma phase over" << std::endl;

	GrB_Matrix Ninv = NULL;
	GrB_Matrix_new(&Ninv, to_grb_real.get_type(), n, s);
	GrB_Matrix_apply(Ninv, NULL, NULL, GrB_MINV_FP32, N, NULL);

	GrB_Matrix BCu = NULL;
	GrB_Matrix_new(&BCu, to_grb_real.get_type(), n, s);
	if (sizeof(RT) == 4)
		GrB_Matrix_assign_FP32(BCu, NULL, NULL, static_cast<RT>(1.0),
							   GrB_ALL, n, GrB_ALL, s, NULL);
	else
		GrB_Matrix_assign_FP64(BCu, NULL, NULL, static_cast<RT>(1.0),
							   GrB_ALL, n, GrB_ALL, s, NULL);

	GrB_Matrix W;
	GrB_Matrix_new(&W, to_grb_real.get_type(), n, s);

	// delta
	for (int32_t d = depth-1; d > 0; --d)
	{
		GrB_Matrix_eWiseMult_BinaryOp
			(W, preds[d], NULL, to_grb_real.get_binary_times(),
			 Ninv, BCu, GrB_DESC_R);

		GrB_mxm(W, preds[d-1], NULL, to_grb_real.get_sr_plus_second(),
				A, W, desc_mxm_02);

		GrB_Matrix_eWiseMult_BinaryOp
			(BCu, NULL, to_grb_real.get_binary_plus(),
			 to_grb_real.get_binary_times(), W, N, NULL);
	}

	// cout << "##delta phase over" << std::endl;


	GrB_Matrix_reduce_BinaryOp(*delta, NULL, NULL,
							 to_grb_real.get_binary_plus(), BCu, NULL);
	if (sizeof(RT) == 4)
		GrB_Vector_apply_BinaryOp2nd_FP32(*delta, NULL, NULL,
										  to_grb_real.get_binary_plus(), *delta,
										  static_cast<RT>(-1.0*s), NULL);
	else
		GrB_Vector_apply_BinaryOp2nd_FP64(*delta, NULL, NULL,
										  to_grb_real.get_binary_plus(), *delta,
										  static_cast<RT>(-1.0*s), NULL);

	// GxB_Vector_fprint(*delta, "delta", GxB_COMPLETE, stdout);


	return;
}



int
main
(
    int    argc,
    char **argv
)
{
	using Value_t = int64_t;
    using Index_t = uint64_t;

	vector<int> tnums;
	if (argc < 4)
	{
        // cout << "Normal usage: ./tricnt-all-grb <matrix-market-file> "
		// 	 << "<numthreads>"
		// 	 << endl;
		tnums = {1, 2, 4, 8, 16, 32, 64};
        // return -1;
    }
	else
	{
        // cout << "Running on " << argv[2] << " processors" << endl << endl;
        // tnums = {atoi(argv[2])};
		tnums = {1, 2, 4, 8, 16, 32, 64};
    }

	
	std::string fileName = getFileName(argv[1]);
	int s = atoi(argv[2]);		

	GrB_init(GrB_BLOCKING);
	GrbAlgObj<Value_t>	to_grb;
	GrB_Matrix			Ain	  = NULL;
	GrB_Index			n, nnz;	
	int nthreads;	
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW); // CSR in GraphBLAS
	GxB_Global_Option_get(GxB_GLOBAL_NTHREADS, &nthreads);
	std::cout << "nthreads (graphblas) " << nthreads << std::endl;


	read_grb_mtx<Value_t>(&Ain, argv[1], true, false, true);
	

	// @formatter:off
    size_t outerIters  = std::getenv("OUTER_ITERS")  ?
		std::stoul(std::getenv("OUTER_ITERS"))  : 1;
    size_t innerIters  = std::getenv("INNER_ITERS")  ?
		std::stoul(std::getenv("INNER_ITERS"))  : 1;
    size_t warmupIters = std::getenv("WARMUP_ITERS") ?
		std::stoul(std::getenv("WARMUP_ITERS")) : (innerIters == 1 ? 0 : 1);
    string mode        = std::getenv("MODE")         ?
		std::getenv("MODE")                         : "";
    // @formatter:on

	if (mode.empty()) { std::cerr << "Mode unspecified!" << std::endl; }
    std::transform(mode.begin(), mode.end(), mode.begin(),
				   [](unsigned char c) { return std::tolower(c); });
	std::cout << "mode:  " << mode << std::endl;

    std::cout << "Iters: " << outerIters << " x (" << warmupIters
			  << "," << innerIters << ")" << std::endl << std::endl;

	GrB_Matrix_nrows(&n, Ain); GrB_Matrix_nvals(&nnz, Ain);
	std::cout << "A: " << n << " " << n << " " << nnz << std::endl;


 	assert(s <= n);
	vector<GrB_Index> srcs;
	for (GrB_Index i = 0; i < n; ++i)
		srcs.push_back(i);
	random_shuffle(srcs.begin(), srcs.end());
	srcs.resize(s);
	GrB_Vector delta = NULL;

	for (size_t i = 0; i < outerIters; i++)
	{
		grb_bc<Index_t, Value_t>("GxB_AxB_DEFAULT", warmupIters, innerIters,
								 tnums, &delta, Ain, srcs);

        // if (mode == "msa" || mode == "all")
		// {
		// 	RUN_CSR(MaskedSPASpGEMM);

        //     RUN_CSR_1P(MSA2A_old);
        //     RUN_CSR_2P(MSA2A_old);
        //     RUN_CSR((MaskedSpGEMM1p<MSA2A<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MSA2A<false, false>::Impl>));

        //     RUN_CSR_1P(MSA1A_old);
        //     RUN_CSR_2P(MSA1A_old);
        //     RUN_CSR((MaskedSpGEMM1p<MSA1A<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MSA1A<false, false>::Impl>));
        // }

        // if (mode == "hash" || mode == "all")
		// {
        //      RUN_CSR(mxm_hash_mask_wobin);
		// 	 RUN_CSR(mxm_hash_mask);
        //      RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, false>::Impl>));
        //      RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, false>::Impl>));
        //  }

        // if (mode == "heap" || mode == "all")
		// {
        //     RUN_CSR_1P(MaskedHeap_v0);
        //     RUN_CSR_2P(MaskedHeap_v0);
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 0>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 0>::Impl>));

        //     RUN_CSR_1P(MaskedHeap_v1);
        //     RUN_CSR_2P(MaskedHeap_v1);
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 1>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 1>::Impl>));

        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 8>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 8>::Impl>));
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 64>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 64>::Impl>));
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 512>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 512>::Impl>));

        //     RUN_CSR_1P(MaskedHeap_v2);
        //     RUN_CSR_2P(MaskedHeap_v2);
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
        // }
		
        // if (mode == "all1p")
		// {
        //     RUN_CSR_CSC(MaskedSpGEMM1p<MaskedInner>);
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 1>::Impl>));
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
        //     RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM1p<MSA1A<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM1p<MSA2A<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM1p<MCA<false, false>::Impl>));
        //     RUN_CSR(MaskedSpGEMM1p);
        // }

        // if (mode == "all2p")
		// {
        //     RUN_CSR_CSC(MaskedSpGEMM2p<MaskedInner>);
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 1>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MSA1A<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MSA2A<false, false>::Impl>));
        //     RUN_CSR((MaskedSpGEMM2p<MCA<false, false>::Impl>));
        // }
	}
	
	
	GrB_Matrix_free(&Ain);
	GrB_finalize();



	return (EXIT_SUCCESS);
}
