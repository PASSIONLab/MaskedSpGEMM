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
grb_ktruss
(
  	const std::string   &inputName,
  	const std::string   &algorithmName,
    const GrB_Matrix	 A,
	size_t				 witers,
	size_t				 niters,
	vector<int>			&tnums,
	int 				 k,
	GrB_Descriptor desc_mxm,
	size_t flops
)
{
	GrB_Index		n, nnz;
	GrbAlgObj<NT>	to_grb;
	GxB_Scalar		s = NULL;

	GxB_Scalar_new(&s, GrB_UINT64);
	GxB_Scalar_setElement_UINT64(s, (uint64_t)k-2);
	GrB_Matrix_nrows(&n, A);

	for (int tnum : tnums)
	{
		GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, tnum);

		GrB_Index nnz_cur, nnz_last;
		int64_t nsteps;

		double t_tot = 0, t_mxm = 0;

		for (int i = 0; i < witers + niters; ++i)
		{
		    double start_iter = omp_get_wtime();
			GrB_Matrix T = NULL;
			GrB_Matrix_nvals(&nnz_last, A);
			nsteps = 1;
			for ( ; ; ++nsteps)
			{
				if (nsteps == 1)
					T = A;

				GrB_Matrix C = NULL;
				GrB_Matrix_new(&C, to_grb.get_type(), n, n);

				double start = omp_get_wtime();
				GrB_mxm(C, T, NULL, to_grb.get_sr_plus_pair(), T, T, desc_mxm);
				double end = omp_get_wtime();
				if (i >= witers) { t_mxm += (end-start); }
				
				GxB_Matrix_select(C, NULL, NULL, GxB_GE_THUNK, C, s, NULL);

				if (nsteps > 1)	// don't clear A
					GrB_Matrix_free(&T);
				T = C;

				GrB_Matrix_nvals(&nnz_cur, C);

				// if (i == 0)
				// 	std::cout << "step " << nsteps
				// 			  << " nnz " << nnz_cur << std::endl;
				if (nnz_last == nnz_cur)
					break;
				nnz_last = nnz_cur;
			}
			GrB_Matrix_free(&T);

			double end_iter = omp_get_wtime();
			if (i >= witers) {
			    t_tot += end_iter - start_iter;
			}

			// if (i == 0)
			// 	std::cout << "[" << tnum << "] number of edges in "
			// 			  << k << "-truss of graph " << nnz_cur
			// 			  << " (" << nsteps << " iterations)" << std::endl;
		}


		t_mxm *= 1e3;
		t_tot *= 1e3;
		t_tot /= (double)niters;
		t_mxm /= (double)niters;

		double mflops = flops / t_mxm / 1e3;

		std::cout << std::setw(12) << "LOG;"
                  << std::setw(20) << getFileName(inputName) << ";"
			      << std::setw(50) << processAlgorithmName(algorithmName) << ";"
                  << std::setw(5) << (std::string(typeid(IT).name()) + "|" + std::string(typeid(NT).name())) << ";"
                  << std::setw(12) << tnum << ";"
                  << std::setw(25) << std::setprecision(4) << std::fixed << t_tot << ";"
                  << std::setw(30) << std::setprecision(4) << std::fixed << t_mxm << ";"
                  << std::setw(30) << std::setprecision(4) << std::fixed << 0.0 << ";"
                  << std::setw(15) << std::setprecision(4) << std::fixed << mflops << ";"
                  << std::setw(10) << nnz_last << ";"
                  << std::setw(10) << nsteps
				  << std::endl;
		
	}


	return;
}



template <class IT, class NT,
		  template<class, class> class AT,
		  template<class, class> class BT,
		  template<class, class> class CT = AT,
		  template<class, class> class MT>
void
msp_ktruss
(
    const std::string &inputName,
	const std::string &algorithmName,
	void(*f)(const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &,
			 const MT<IT, NT> &, NT(NT&, NT&), plus<NT>, unsigned),
	size_t		 witers,
	size_t		 niters,
	vector<int> &tnums,
	GrB_Matrix	 A,
	int			 k,
	size_t flops
)
{
	GrB_Index		n, nnz;
	GrbAlgObj<NT>	to_grb;
	GxB_Scalar		s = NULL;
	auto			f_one = [] (NT& arg1, NT& arg2) -> NT {return (NT) 1;};

	GxB_Scalar_new(&s, GrB_UINT64);
	GxB_Scalar_setElement_UINT64(s, (uint64_t)k-2);
	GrB_Matrix_nrows(&n, A);

	for (int tnum : tnums)
	{
		GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, tnum);

		GrB_Index nnz_cur, nnz_last;
		int64_t nsteps;

		double t_tot = 0, t_mxm = 0, t_tran = 0;

		for (int i = 0; i < witers + niters; ++i)
		{
		    double start_iter = omp_get_wtime();

		    GrB_Matrix T = NULL;
			GrB_Matrix_nvals(&nnz_last, A);
			nsteps = 1;
			for ( ; ; ++nsteps)
			{
				if (nsteps == 1)
					T = A;

				GrB_Matrix C = NULL;
				GrB_Matrix_new(&C, to_grb.get_type(), n, n);

				GrB_Matrix T2;
				if constexpr(!std::is_same_v<AT<IT, NT>, BT<IT, NT>>) {
				    double start_tran = omp_get_wtime();
				    GxB_Format_Value old;
				    GxB_Global_Option_get(GxB_FORMAT, &old);
				    GxB_Global_Option_set(GxB_FORMAT, GxB_BY_COL);
				    GrB_Matrix_new(&T2, to_grb.get_type(), n, n);
				    GrB_transpose(T2, GrB_NULL, GrB_NULL, T, GrB_DESC_T0);
				    GxB_Global_Option_set(GxB_FORMAT, old);
                    double end_tran = omp_get_wtime();
                    if (i >= witers) { t_tran += (end_tran - start_tran); }
				}

				AT<IT, NT> T1_msp(T); CT<IT, NT> C_msp(C);

				double start = omp_get_wtime();
				if constexpr(std::is_same_v<AT<IT, NT>, BT<IT, NT>>) {
				    f(T1_msp, T1_msp, C_msp, T1_msp, f_one, plus<NT>(), tnum);
				} else {
                    BT<IT, NT> T2_msp = BT<IT, NT>(T2);
                    f(T1_msp, T2_msp, C_msp, T1_msp, f_one, plus<NT>(), tnum);
                    T2_msp.get_grb_mat(T2);
                    GrB_Matrix_free(&T2);
				}
				double end = omp_get_wtime();
				if (i >= witers) { t_mxm += (end-start); }

				C_msp.get_grb_mat(C); T1_msp.get_grb_mat(T);
				
				GxB_Matrix_select(C, NULL, NULL, GxB_GE_THUNK, C, s, NULL);

				if (nsteps > 1)	// don't clear A
					GrB_Matrix_free(&T);
				T = C;

				GrB_Matrix_nvals(&nnz_cur, C);

				// if (i == 0)
				// 	std::cout << "step " << nsteps
				// 			  << " nnz " << nnz_cur << std::endl;
				if (nnz_last == nnz_cur)
					break;
				nnz_last = nnz_cur;
			}
			GrB_Matrix_free(&T);

			double end_iter = omp_get_wtime();
			if (i >= witers) {
			    t_tot += end_iter - start_iter;
			}

			// if (i == 0)
			// 	std::cout << "[" << tnum << "] number of edges in "
			// 			  << k << "-truss of graph " << nnz_cur
			// 			  << " (" << nsteps << " iterations)" << std::endl;
		}


		t_mxm *= 1e3;
		t_tot *= 1e3;
		t_tran *= 1e3;
		t_tot /= (double)niters;
		t_mxm /= (double)niters;
		t_tran /= (double)niters;

		double mflops = flops / (t_mxm + t_tran) / 1e3;

		std::cout << std::setw(12) << "LOG;"
                  << std::setw(20) << getFileName(inputName) << ";"
			      << std::setw(50) << processAlgorithmName(algorithmName) << ";"
                  << std::setw(5) << (std::string(typeid(IT).name()) + "|" + std::string(typeid(NT).name())) << ";"
                  << std::setw(12) << tnum << ";"
                  << std::setw(25) << std::setprecision(4) << std::fixed << t_tot << ";"
                  << std::setw(30) << std::setprecision(4) << std::fixed << t_mxm << ";"
                  << std::setw(30) << std::setprecision(4) << std::fixed << t_tran << ";"
                  << std::setw(15) << std::setprecision(4) << std::fixed << mflops << ";"
                  << std::setw(10) << nnz_last << ";"
                  << std::setw(10) << nsteps
				  << std::endl;
		
	}


	return;
}

template <class IT, class NT,
		  template<class, class> class AT,
		  template<class, class> class BT,
		  template<class, class> class CT = AT,
		  template<class, class> class MT>
size_t
ktruss_mult_ops
(
	void(*f)(const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &,
			 const MT<IT, NT> &, NT(NT&, NT&), plus<NT>, unsigned),
	GrB_Matrix	 A,
	int			 k
)
{
	GrB_Index		n, nnz;
	GrbAlgObj<NT>	to_grb;
	GxB_Scalar		s = NULL;
	auto			f_one = [] (NT& arg1, NT& arg2) -> NT {return (NT) 1;};
    size_t          multOps = 0;
    int             numThreads = omp_get_max_threads();


	GxB_Scalar_new(&s, GrB_UINT64);
	GxB_Scalar_setElement_UINT64(s, (uint64_t)k-2);
	GrB_Matrix_nrows(&n, A);

    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, numThreads);

    GrB_Index nnz_cur, nnz_last;
    int64_t nsteps;
    double start_iter = omp_get_wtime();

    GrB_Matrix T = NULL;
    GrB_Matrix_nvals(&nnz_last, A);
    nsteps = 1;
    for ( ; ; ++nsteps)
    {
        if (nsteps == 1)
            T = A;

        GrB_Matrix C = NULL;
        GrB_Matrix_new(&C, to_grb.get_type(), n, n);

        AT<IT, NT> T1_msp(T); CT<IT, NT> C_msp(C);

        auto nops = calculateMultOps(T1_msp, T1_msp, numThreads);
        multOps += nops;
        f(T1_msp, T1_msp, C_msp, T1_msp, f_one, plus<NT>(), numThreads);

        C_msp.get_grb_mat(C); T1_msp.get_grb_mat(T);

        GxB_Matrix_select(C, NULL, NULL, GxB_GE_THUNK, C, s, NULL);

        if (nsteps > 1)	// don't clear A
            GrB_Matrix_free(&T);
        T = C;

        GrB_Matrix_nvals(&nnz_cur, C);

        if (nnz_last == nnz_cur)
            break;
        nnz_last = nnz_cur;
    }
    GrB_Matrix_free(&T);

	return multOps;
}



#define RUN_CSR_IMPL(NAME, FUNC) msp_ktruss<Index_t, Value_t, CSR, CSR, CSR, CSR>(fileName, NAME, FUNC, warmupIters, innerIters, tnums, Ain, k, flop)
#define RUN_CSR(ALG) RUN_CSR_IMPL(#ALG, ALG)
#define RUN_CSR_1P(ALG) RUN_CSR_IMPL(#ALG "-1P", MaskedSpGEMM1p<ALG>)
#define RUN_CSR_2P(ALG) RUN_CSR_IMPL(#ALG "-2P", MaskedSpGEMM2p<ALG>)

#define RUN_CSR_CSC_IMPL(NAME, FUNC) msp_ktruss<Index_t, Value_t, CSR, CSC, CSR, CSR>(fileName, NAME, FUNC, warmupIters, innerIters, tnums, Ain, k, flop)
#define RUN_CSR_CSC(ALG) RUN_CSR_CSC_IMPL(#ALG, ALG)


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
        for (int i = 3; i < argc; i++) {
            tnums.emplace_back(atoi(argv[i]));
        }
    }


	std::string fileName = getFileName(argv[1]);
	int k = atoi(argv[2]);		// k-truss
	assert(k >= 2);
		
	GrB_init(GrB_BLOCKING);
	GrbAlgObj<Value_t>	to_grb;
	GrB_Matrix			Ain	  = NULL;
	GrB_Index			n, nnz;	
	int nthreads;

	double bswitch [GxB_NBITMAP_SWITCH];
	std::fill(bswitch, bswitch + GxB_NBITMAP_SWITCH, 1.0);
	GxB_Global_Option_set(GxB_BITMAP_SWITCH, bswitch);
	GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW); // CSR in GraphBLAS
	GxB_Global_Option_get(GxB_GLOBAL_NTHREADS, &nthreads);
	std::cout << "nthreads (graphblas) " << nthreads << std::endl;

	read_grb_mtx<Value_t>(&Ain, argv[1], true, true, true);


	// @formatter:off
    size_t outerIters  = std::getenv("OUTER_ITERS")  ?
		std::stoul(std::getenv("OUTER_ITERS"))  : 1;
    size_t innerIters  = std::getenv("INNER_ITERS")  ?
		std::stoul(std::getenv("INNER_ITERS"))  : 1;
    size_t warmupIters = std::getenv("WARMUP_ITERS") ?
		std::stoul(std::getenv("WARMUP_ITERS")) : (innerIters == 1 ? 0 : 1);
    string mode        = std::getenv("MODE")         ?
		std::getenv("MODE")                         : "";
    bool disableHeap   = std::getenv("DISABLE_HEAP") ? true : false;
    // @formatter:on

	if (mode.empty()) { std::cerr << "Mode unspecified!" << std::endl; }
    std::transform(mode.begin(), mode.end(), mode.begin(),
				   [](unsigned char c) { return std::tolower(c); });
	std::cout << "mode:  " << mode << std::endl;

    std::cout << "Iters: " << outerIters << " x (" << warmupIters
			  << ";" << innerIters << ")" << std::endl << std::endl;

	GrB_Matrix_nrows(&n, Ain); GrB_Matrix_nvals(&nnz, Ain);
	std::cout << "A: " << n << " " << n << " " << nnz << std::endl;

	auto flop = ktruss_mult_ops<Index_t, Value_t, CSR, CSR, CSR, CSR>
	        (MaskedSpGEMM1p<MSA2A<false, false>::Impl>, Ain, k) * 2;

	if (mode == "flops") {
	    std::cout << "LOG;" <<fileName << ";" << flop << std::endl;
	}

	std::cout << std::setw(12) << "LOG-header;"
	<< std::setw(20) << "FileName" << ";"
	<< std::setw(50) << "Algorithm" << ";"
	<< std::setw(5) << "Type" << ";"
	<< std::setw(12) << "NumThreads" << ";"
	<< std::setw(25) << "AverageTime-total(ms)" << ";"
	<< std::setw(30) << "AverageTime-mxm(ms)" << ";"
	<< std::setw(30) << "AverageTime-Transpose(ms)" << ";"
	<< std::setw(15) << "MFLOPS" << ";"
	<< std::setw(10) << "nnz-last" << ";"
	<< std::setw(10) << "nsteps" << std::endl;

	for (size_t i = 0; i < outerIters; i++)
	{
	    if (mode == "grb" || mode == "all" || mode == "benchmark") {
	        GrB_Descriptor desc_mxm = NULL;
	        GrB_Descriptor_new(&desc_mxm);
	        GxB_Desc_set(desc_mxm, GrB_MASK, GrB_STRUCTURE);

	        for (const auto &method : std::vector<std::tuple<
	        GrB_Desc_Value, std::string, GrB_Matrix>>
	        ({
	            {GxB_DEFAULT,       "DEFAULT",       Ain},
	            {GxB_AxB_DOT,       "DOT-TRANSPOSE", Ain},
	            {GxB_AxB_SAXPY,     "SAXPY",         Ain},
	            {GxB_AxB_GUSTAVSON, "GUSTAVSON",     Ain},
	            {GxB_AxB_HASH,      "HASH",          Ain},
	            })) {
	            std::string name = "GxB_AxB_" + std::get<1>(method);
	            GxB_Desc_set(desc_mxm, GxB_AxB_METHOD, std::get<0>(method));
	            //                GxB_Desc_set(desc_mxm, GxB_SORT, 0);
	            //                grb_tri_count_sandia_L<Index_t, Value_t>
	            //                        (fileName, name, std::get<2>(method), std::get<3>(method),
	            //                         warmupIters, innerIters, tnums, flop, desc_mxm);

	            name += "-sorted";
	            GxB_Desc_set(desc_mxm, GxB_SORT, 1); // want output sorted
	            grb_ktruss<Index_t, Value_t>
	            (fileName, name, std::get<2>(method),
                 warmupIters, innerIters, tnums, k, desc_mxm, flop);
	        }
	    }

        if (mode == "msa" || mode == "all")
		{
			RUN_CSR(MaskedSPASpGEMM);

            RUN_CSR_1P(MSA2A_old);
            RUN_CSR_2P(MSA2A_old);
            RUN_CSR((MaskedSpGEMM1p<MSA2A<false, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MSA2A<false, false>::Impl>));

            RUN_CSR_1P(MSA1A_old);
            RUN_CSR_2P(MSA1A_old);
            RUN_CSR((MaskedSpGEMM1p<MSA1A<false, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MSA1A<false, false>::Impl>));
        }

        if (mode == "hash" || mode == "all")
		{
             RUN_CSR(mxm_hash_mask_wobin);
			 RUN_CSR(mxm_hash_mask);
             RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, false>::Impl>));
        }

        if (mode == "heap" || mode == "all")
		{
            RUN_CSR_1P(MaskedHeap_v0);
            RUN_CSR_2P(MaskedHeap_v0);
            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 0>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 0>::Impl>));

            RUN_CSR_1P(MaskedHeap_v1);
            RUN_CSR_2P(MaskedHeap_v1);
            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 1>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 1>::Impl>));

            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 8>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 8>::Impl>));
            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 64>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 64>::Impl>));
            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 512>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 512>::Impl>));

            RUN_CSR_1P(MaskedHeap_v2);
            RUN_CSR_2P(MaskedHeap_v2);
            RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
        }
		
         if (mode == "all1p")
		 {
             RUN_CSR_CSC(MaskedSpGEMM1p<MaskedInner>);
             RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 1>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MSA1A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MSA2A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MCA<false, false>::Impl>));
             RUN_CSR(MaskedSpGEMM1p);
         }

         if (mode == "all2p")
		 {
             RUN_CSR_CSC(MaskedSpGEMM2p<MaskedInner>);
             RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 1>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MSA1A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MSA2A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MCA<false, false>::Impl>));
         }

        if (mode == "benchmark") {
            if (!disableHeap) {
                RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 1>::Impl>));
                RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
            }

            RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, false>::Impl>));
            RUN_CSR((MaskedSpGEMM1p<MSA2A<false, false>::Impl>));
            RUN_CSR((MaskedSpGEMM1p<MCA<false, false>::Impl>));
            RUN_CSR_CSC(MaskedSpGEMM1p<MaskedInner>);

            if (!disableHeap) {
                RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 1>::Impl>));
                RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
            }
            RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MSA2A<false, false>::Impl>));
            RUN_CSR((MaskedSpGEMM2p<MCA<false, false>::Impl>));
            RUN_CSR_CSC(MaskedSpGEMM2p<MaskedInner>);
        }
	}
	
	
	GrB_Matrix_free(&Ain);
	GrB_finalize();



	return (EXIT_SUCCESS);
}
