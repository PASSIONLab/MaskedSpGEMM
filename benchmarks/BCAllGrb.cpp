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
	GrB_Index			n, s, nnz;
	GrbAlgObj<NT>		to_grb;
	GrB_Matrix			F  = NULL, N = NULL, Atr = NULL;
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
	assert(s > 0);
	GrB_Matrix_nrows(&n, A);
	GrB_Vector_new(delta, to_grb.get_type(), n);

	vector<GrB_Index> cids(s);
	std::iota(cids.begin(), cids.end(), 0);

	// frontier and shortest path matrices
	vector<NT> vals(s, 1.0);
	GrbMatrixBuild<NT>()(&F, srcs.data(), cids.data(), vals.data(), n, s, s);	
	avoid_iso(F, static_cast<NT>(1), srcs[0], cids[0]);	
	GrB_Matrix_dup(&N, F);
	GrB_Matrix_new(&Atr, to_grb.get_type(), n, n);
	GrB_transpose(Atr, NULL, NULL, A, NULL);

	// sigma
	GrB_Index	nvals = s;
	int32_t		depth = 0;
	while (nvals > 0)
	{
		GrB_mxm(F, N, NULL, to_grb.get_sr_plus_second(), Atr, F, desc_mxm_01);

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
	GrB_Matrix_new(&Ninv, to_grb.get_type(), n, s);
	GrB_Matrix_apply(Ninv, NULL, NULL, GrB_MINV_FP32, N, NULL);

	GrB_Matrix BCu = NULL;
	GrB_Matrix_new(&BCu, to_grb.get_type(), n, s);
	if (sizeof(NT) == 4)
		GrB_Matrix_assign_FP32(BCu, NULL, NULL, static_cast<NT>(1.0),
							   GrB_ALL, n, GrB_ALL, s, NULL);
	else
		GrB_Matrix_assign_FP64(BCu, NULL, NULL, static_cast<NT>(1.0),
							   GrB_ALL, n, GrB_ALL, s, NULL);

	avoid_iso(BCu, static_cast<NT>(1.0), 0, 0);

	GrB_Matrix W;
	GrB_Matrix_new(&W, to_grb.get_type(), n, s);

	// delta
	for (int32_t d = depth-1; d > 0; --d)
	{
		GrB_Matrix_eWiseMult_BinaryOp
			(W, preds[d], NULL, to_grb.get_binary_times(),
			 Ninv, BCu, GrB_DESC_R);

		GrB_mxm(W, preds[d-1], NULL, to_grb.get_sr_plus_second(),
				A, W, desc_mxm_02);

		GrB_Matrix_eWiseMult_BinaryOp
			(BCu, NULL, to_grb.get_binary_plus(),
			 to_grb.get_binary_times(), W, N, NULL);
	}

	// cout << "##delta phase over" << std::endl;


	GrB_Matrix_reduce_BinaryOp(*delta, NULL, NULL,
							 to_grb.get_binary_plus(), BCu, NULL);
	if (sizeof(NT) == 4)
		GrB_Vector_apply_BinaryOp2nd_FP32(*delta, NULL, NULL,
										  to_grb.get_binary_plus(), *delta,
										  static_cast<NT>(-1.0*s), NULL);
	else
		GrB_Vector_apply_BinaryOp2nd_FP64(*delta, NULL, NULL,
										  to_grb.get_binary_plus(), *delta,
										  static_cast<NT>(-1.0*s), NULL);

	GxB_Vector_fprint(*delta, "delta", GxB_COMPLETE, stdout);


	return;
}



template <class IT,
		  class NT,
		  template<class, class> class AT,
		  template<class, class> class BT,
		  template<class, class> class CT = AT,
		  template<class, class> class MT>
void
msp_bc
(
  	const std::string	&inputName,
	const std::string 	&algorithmName,
	void(*f_plain)(const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &,
				   const MT<IT, NT> &,
				   NT(NT&, NT&),
				   // multiplies<NT>,
				   plus<NT>, unsigned),
	void(*f_cmpl)(const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &,
				  const MT<IT, NT> &,
				  NT(NT&, NT&),
				  // multiplies<NT>,
				  plus<NT>, unsigned),
	size_t				 witers,
	size_t				 niters,
	vector<int>			&tnums,
	GrB_Vector			*delta,
	GrB_Matrix			 A,
	vector<GrB_Index>	&srcs
)
{
	static_assert(std::is_same<NT, float>::value ||
				  std::is_same<NT, double>::value,
				  "Matrix must be real type for betweenness centrality");
	GrB_Index			n, s, nnz;
	GrbAlgObj<NT>		to_grb;
	GrB_Matrix			F  = NULL, N = NULL, Atr = NULL;
	vector<GrB_Matrix>	preds;

	auto f_second = [] (NT &arg1, NT &arg2) -> NT {return arg2;};

	s = srcs.size();
	assert(s > 0);
	GrB_Matrix_nrows(&n, A);
	GrB_Vector_new(delta, to_grb.get_type(), n);

	vector<GrB_Index> cids(s);
	std::iota(cids.begin(), cids.end(), 0);

	// frontier and shortest path matrices
	vector<NT> vals(s, 1.0);
	GrbMatrixBuild<NT>()(&F, srcs.data(), cids.data(), vals.data(), n, s, s);	
	avoid_iso(F, static_cast<NT>(1), srcs[0], cids[0]);	
	GrB_Matrix_dup(&N, F);
	GrB_Matrix_new(&Atr, to_grb.get_type(), n, n);
	GrB_transpose(Atr, NULL, NULL, A, NULL);

	int tnum = 1;

	AT<IT, NT> Atr_msp(Atr);

	// sigma
	GrB_Index	nvals = s;
	int32_t		depth = 0;
	while (nvals > 0)
	{
		BT<IT, NT> F_msp(F); MT<IT, NT> N_msp(N);
		f_cmpl(Atr_msp, F_msp, F_msp, N_msp, f_second, plus<NT>(), tnum);
		F_msp.get_grb_mat(F); N_msp.get_grb_mat(N);

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
	GrB_Matrix_new(&Ninv, to_grb.get_type(), n, s);
	GrB_Matrix_apply(Ninv, NULL, NULL, GrB_MINV_FP32, N, NULL);

	GrB_Matrix BCu = NULL;
	GrB_Matrix_new(&BCu, to_grb.get_type(), n, s);
	if (sizeof(NT) == 4)
		GrB_Matrix_assign_FP32(BCu, NULL, NULL, static_cast<NT>(1.0),
							   GrB_ALL, n, GrB_ALL, s, NULL);
	else
		GrB_Matrix_assign_FP64(BCu, NULL, NULL, static_cast<NT>(1.0),
							   GrB_ALL, n, GrB_ALL, s, NULL);
	
	avoid_iso(BCu, static_cast<NT>(1.0), 0, 0);

	GrB_Matrix W;
	GrB_Matrix_new(&W, to_grb.get_type(), n, s);
	
	AT<IT, NT> A_msp(A);

	// delta
	for (int32_t d = depth-1; d > 0; --d)
	{
		GrB_Matrix_eWiseMult_BinaryOp
			(W, preds[d], NULL, to_grb.get_binary_times(),
			 Ninv, BCu, GrB_DESC_R);

		BT<IT, NT> W_msp(W); MT<IT, NT> Pred_msp(preds[d-1]);
		f_plain(A_msp, W_msp, W_msp, Pred_msp, f_second, plus<NT>(), tnum);		
		W_msp.get_grb_mat(W); Pred_msp.get_grb_mat(preds[d-1]);

		GrB_Matrix_eWiseMult_BinaryOp
			(BCu, NULL, to_grb.get_binary_plus(),
			 to_grb.get_binary_times(), W, N, NULL);
	}

	// cout << "##delta phase over" << std::endl;


	GrB_Matrix_reduce_BinaryOp(*delta, NULL, NULL,
							 to_grb.get_binary_plus(), BCu, NULL);
	if (sizeof(NT) == 4)
		GrB_Vector_apply_BinaryOp2nd_FP32(*delta, NULL, NULL,
										  to_grb.get_binary_plus(), *delta,
										  static_cast<NT>(-1.0*s), NULL);
	else
		GrB_Vector_apply_BinaryOp2nd_FP64(*delta, NULL, NULL,
										  to_grb.get_binary_plus(), *delta,
										  static_cast<NT>(-1.0*s), NULL);

	GxB_Vector_fprint(*delta, "delta", GxB_COMPLETE, stdout);


	A_msp.get_grb_mat(A);

	return;
}



#define RUN_CSR_IMPL(NAME, FUNC_PLAIN, FUNC_CMPL) msp_bc<Index_t, Value_t, CSR, CSR, CSR, CSR>(fileName, NAME, FUNC_PLAIN, FUNC_CMPL, warmupIters, innerIters, tnums, &delta, Ain, srcs)
#define RUN_CSR(ALG_PLAIN, ALG_CMPL) RUN_CSR_IMPL(#ALG_PLAIN, ALG_PLAIN, ALG_CMPL)
#define RUN_CSR_1P(ALG) RUN_CSR_IMPL(#ALG "-1P", MaskedSpGEMM1p<ALG>)
#define RUN_CSR_2P(ALG) RUN_CSR_IMPL(#ALG "-2P", MaskedSpGEMM2p<ALG>)



int
main
(
    int    argc,
    char **argv
)
{
	using Value_t = float;
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
	float x = atof(argv[2]);
	assert(x >= 0.0 && x <= 1.0);
	

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

	int s = round(static_cast<float>(n)*x);
 	assert(s > 0 && s <= n);
	vector<GrB_Index> srcs;
	for (GrB_Index i = 0; i < n; ++i)
		srcs.push_back(i);
	random_shuffle(srcs.begin(), srcs.end());
	srcs.resize(s);
	// srcs = {0,1,2,3,4,5,6,7};
	GrB_Vector delta = NULL;

	for (size_t i = 0; i < outerIters; i++)
	{
		grb_bc<Index_t, Value_t>("GxB_AxB_DEFAULT", warmupIters, innerIters,
								 tnums, &delta, Ain, srcs);

		if (mode == "heap" || mode == "all") {
            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 0>::Impl>),
					(MaskedSpGEMM1p<MaskedHeap<true, true, 0>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 0>::Impl>),
					(MaskedSpGEMM2p<MaskedHeap<true, true, 0>::Impl>));
        }

        if (mode == "hash" || mode == "all") {
            RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, false>::Impl>),
					(MaskedSpGEMM1p<MaskedHash<true, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, false>::Impl>),
					(MaskedSpGEMM2p<MaskedHash<true, false>::Impl>));

            // RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, true>::Impl>),
			// 		(MaskedSpGEMM1p<MaskedHash<true, true>::Impl>));
            // RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, true>::Impl>),
			// 		(MaskedSpGEMM2p<MaskedHash<true, true>::Impl>));
        }

        if (mode == "msa" || mode == "all") {
            RUN_CSR((MaskedSpGEMM1p<MSA1A<false, false>::Impl>),
					(MaskedSpGEMM1p<MSA1A<true, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MSA1A<false, false>::Impl>),
					(MaskedSpGEMM2p<MSA1A<true, false>::Impl>));

            // RUN_CSR((MaskedSpGEMM1p<MSA1A<false, true>::Impl>),
			// 		(MaskedSpGEMM1p<MSA1A<true, true>::Impl>));
            // RUN_CSR((MaskedSpGEMM2p<MSA1A<false, true>::Impl>),
			// 		(MaskedSpGEMM2p<MSA1A<true, true>::Impl>));

            RUN_CSR((MaskedSpGEMM1p<MSA2A<false, false>::Impl>),
					(MaskedSpGEMM1p<MSA2A<true, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MSA2A<false, false>::Impl>),
					(MaskedSpGEMM2p<MSA2A<true, false>::Impl>));

            // RUN_CSR((MaskedSpGEMM1p<MSA2A<false, true>::Impl>),
			// 		(MaskedSpGEMM1p<MSA2A<true, true>::Impl>));
            // RUN_CSR((MaskedSpGEMM2p<MSA2A<true, true>::Impl>),
			// 		(MaskedSpGEMM2p<MSA2A<false, true>::Impl>));
        }		
	}
	
	
	GrB_Matrix_free(&Ain);
	GrB_finalize();



	return (EXIT_SUCCESS);
}
