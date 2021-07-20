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
grb_tri_count_sandia_L
(
    const std::string   &inputName,
    const std::string   &algorithmName,
    const GrB_Matrix	 L,
	size_t				 witers,
	size_t				 niters,
	vector<int>			&tnums,
	size_t				 nfop,
	GrB_Descriptor		 desc_mxm	
)
{
	GrB_Index		n, nnz;
	GrbAlgObj<NT>	to_grb;	
	GrB_Matrix		C	  = NULL;

	GrB_Matrix_nrows(&n, L);

	for (int tnum : tnums)
	{
		GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, tnum);

        GrB_Matrix_new(&C, to_grb.get_type(), n, n);

        for (int i = 0; i < witers; ++i)
		{
            GrB_mxm (C, L, NULL, to_grb.get_sr_plus_pair(), L, L, desc_mxm);
        }

        double ave_msec = 0;
        for (int i = 0; i < niters; ++i)
		{
			GrB_Matrix_clear(C);

            double start = omp_get_wtime();
            GrB_mxm(C, L, NULL, to_grb.get_sr_plus_pair(), L, L, desc_mxm);
			double end = omp_get_wtime();

            double msec = (end - start) * 1000;
            ave_msec += msec;
        }
		
        ave_msec /= double(niters);
        double mflops = (double) nfop / ave_msec / 1000;
		uint64_t ntri = 0;
		GrB_Matrix_reduce_UINT64(&ntri, NULL,
								 to_grb.get_monoid_plus(), C, NULL);
		GrB_Matrix_nvals(&nnz, C);

        std::cout << std::setw(12) << "LOG;"
                  << std::setw(20) << getFileName(inputName) << ";"
                  << std::setw(50) << processAlgorithmName(algorithmName) << ";"
                  << std::setw(5) << (std::string(typeid(IT).name()) + "|" + std::string(typeid(NT).name())) << ";"
                  << std::setw(12) << tnum << ";"
                  << std::setw(20) << std::setprecision(4) << std::fixed << ave_msec << ";"
                  << std::setw(15) << std::setprecision(4) << std::fixed << mflops << ";"
                  << std::setw(10) << nnz << ";"
                  << std::setw(10) << ntri << ";"
                  << std::endl;
		
        GrB_Matrix_clear(C);
    }
}



template <class IT, class NT,
		  template<class, class> class AT,
		  template<class, class> class BT,
		  template<class, class> class CT = AT,
		  template<class, class> class MT>
void
msp_tri_count_sandia_L
(
    const std::string &inputName,
	const std::string &algorithmName,
	void(*f)(const AT<IT, NT> &, const BT<IT, NT> &, CT<IT, NT> &,
			 const MT<IT, NT> &,
			 NT(NT&, NT&),
			 // multiplies<NT>,
			 plus<NT>, unsigned),
	size_t witers,
	size_t niters,
	vector<int> &tnums,
	size_t nflop,
	GrB_Matrix L
)
{
	GrB_Index		n, nnz;
	GrbAlgObj<NT>	to_grb;	
	GrB_Matrix		C	   = NULL;
	auto			f_one = [] (NT &arg1, NT &arg2) -> NT {return (NT) 1;};
	AT<IT, NT>		L_msp(L);
	

	GrB_Matrix_nrows(&n, L);
	
    for (int tnum : tnums)
	{
        omp_set_num_threads(tnum);

        CT<IT, NT> C_msp;
        
        for (int i = 0; i < witers; ++i)
		{
			f(L_msp, L_msp, C_msp, L_msp, f_one, plus<NT>(), tnum);
		}

        double ave_msec = 0;
        for (int i = 0; i < niters; ++i)
		{
            C_msp.make_empty();

            double start = omp_get_wtime();
            f(L_msp, L_msp, C_msp, L_msp, f_one, plus<NT>(), tnum);
			double end = omp_get_wtime();			

            double msec = (end - start) * 1000;
            ave_msec += msec;
        }

        ave_msec /= static_cast<double>(niters);
        double mflops = (double) nflop / ave_msec / 1000;
		
		GrB_Matrix_new(&C, to_grb.get_type(), n, n);
		C_msp.get_grb_mat(C);
		uint64_t ntri = 0;
		GrB_Matrix_reduce_UINT64(&ntri, NULL,
								 to_grb.get_monoid_plus(), C, NULL);
		GrB_Matrix_nvals(&nnz, C);

        std::cout << std::setw(12) << "LOG;"
                  << std::setw(20) << getFileName(inputName) << ";"
                  << std::setw(50) << processAlgorithmName(algorithmName) << ";"
                  << std::setw(5) << (std::string(typeid(IT).name()) + "|" + std::string(typeid(NT).name())) << ";"
                  << std::setw(12) << tnum << ";"
                  << std::setw(20) << std::setprecision(4) << std::fixed << ave_msec << ";"
                  << std::setw(15) << std::setprecision(4) << std::fixed << mflops << ";"
				  << std::setw(10) << nnz << ";"
                  << std::setw(10) << ntri << ";"
				  << std::endl;

		GrB_Matrix_clear(C);
    }

	
	L_msp.get_grb_mat(L);		// restore L
}



#define RUN_CSR_IMPL(NAME, FUNC) msp_tri_count_sandia_L<Index_t, Value_t, CSR, CSR, CSR, CSR>(fileName, NAME, FUNC, warmupIters, innerIters, tnums, flop, L)
#define RUN_CSR(ALG) RUN_CSR_IMPL(#ALG, ALG)
#define RUN_CSR_1P(ALG) RUN_CSR_IMPL(#ALG "-1P", MaskedSpGEMM1p<ALG>)
#define RUN_CSR_2P(ALG) RUN_CSR_IMPL(#ALG "-2P", MaskedSpGEMM2p<ALG>)



int
main (int argc,
	  char **argv)
{
	using Value_t = int64_t;
    using Index_t = uint64_t;

	vector<int> tnums;	
	if (argc < 3)
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
        for (int i = 2; i < argc; i++) {
            tnums.emplace_back(atoi(argv[i]));
        }
    }

	// tnums = {1};

	std::string fileName = getFileName(argv[1]);

	GrB_init(GrB_NONBLOCKING);
	GrbAlgObj<Value_t>	to_grb;
	GrB_Matrix			Ain	  = NULL, L = NULL;
	GrB_Index			n, nnz;	
	int nthreads;

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
    // @formatter:on

	if (mode.empty()) { std::cerr << "Mode unspecified!" << std::endl; }
    std::transform(mode.begin(), mode.end(), mode.begin(),
				   [](unsigned char c) { return std::tolower(c); });
	std::cout << "mode:  " << mode << std::endl;

    std::cout << "Iters: " << outerIters << " x (" << warmupIters
			  << "," << innerIters << ")" << std::endl << std::endl;

	GrB_Matrix_nrows(&n, Ain); GrB_Matrix_nvals(&nnz, Ain);
	std::cout << "A: " << n << " " << n << " " << nnz << std::endl;
	get_lowtri<Value_t>(Ain, &L);
	GrB_Matrix_free(&Ain);
	GrB_Matrix_nrows(&n, L); GrB_Matrix_nvals(&nnz, L);
	std::cout << "L: " << n << " " << n << " " << nnz << std::endl;

	CSR<Index_t, Value_t> tmp(L);
	std::size_t flop = get_flop(tmp, tmp);
	tmp.get_grb_mat(L);
	
	// GxB_Matrix_fprint(L, "L", GxB_SUMMARY, stdout);

    std::cout << std::setw(12) << "LOG-header;"
              << std::setw(20) << "File name" << ";"
              << std::setw(50) << "Algorithm" << ";"
              << std::setw(5) << "Type" << ";"
              << std::setw(12) << "NumThreads" << ";"
              << std::setw(20) << "Average time (s)" << ";"
              << std::setw(15) << "MFLOPS" << ";"
              << std::setw(10) << "C-nvals" << ";"
              << std::setw(10) << "C-sum" << std::endl;

	for (size_t i = 0; i < outerIters; i++)
	{
		// GraphBLAS only
		GrB_Descriptor desc_mxm = NULL;
		GrB_Descriptor_new(&desc_mxm);
        GxB_Desc_set(desc_mxm, GrB_MASK, GrB_STRUCTURE);

        grb_tri_count_sandia_L<Index_t, Value_t>
                (fileName, "GxB_AxB_DEFAULT", L, warmupIters, innerIters,
                 tnums, flop, desc_mxm);

		GxB_Desc_set(desc_mxm, GxB_SORT, 1); // want output sorted
        grb_tri_count_sandia_L<Index_t, Value_t>
                (fileName, "GxB_AxB_DEFAULT-Sorted", L, warmupIters, innerIters,
                 tnums, flop, desc_mxm);

//		 if (mode == "inner" || mode == "dot" || mode == "all")
//		 {
//             RUN_CSR_CSC((innerSpGEMM_nohash<false, false>));
//             RUN_CSR_CSC(MaskedSpGEMM1pInnerProduct);
//             RUN_CSR_CSC(MaskedSpGEMM2pInnerProduct);
//             RUN_CSR_CSC(MaskedSpGEMM1p<MaskedInner>);
//             RUN_CSR_CSC(MaskedSpGEMM2p<MaskedInner>);
//         }

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
//             RUN_CSR_CSC(MaskedSpGEMM1p<MaskedInner>);
             RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, 1>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MaskedHash<false, false>::Impl>));
//             RUN_CSR((MaskedSpGEMM1p<MSA1A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MSA2A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM1p<MCA<false, false>::Impl>));
//             RUN_CSR(MaskedSpGEMM1p);
         }

         if (mode == "all2p")
		 {
//             RUN_CSR_CSC(MaskedSpGEMM2p<MaskedInner>);
             RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, 1>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MaskedHeap<false, true, MaskedHeapDot>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MaskedHash<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MSA1A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MSA2A<false, false>::Impl>));
             RUN_CSR((MaskedSpGEMM2p<MCA<false, false>::Impl>));
         }
	}

	
	// GrB_Matrix_free(&L);
	GrB_finalize();
	
	return (EXIT_SUCCESS);
}
