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

//#include "overridenew.h"
#include "utility.h"
#include "CSC.h"
#include "CSR.h"
#include "IO.h"
#include "multiply.h"

#ifdef KNL_EXE
#include "hash_mult.h"
#elif defined HW_EXE
#include "hash_mult_hw.h"
#else
#include "hash_mult_hw.h"
#endif

#include "bloom_filter.hpp"
using namespace std;

extern "C" {
#include <mkl_spblas.h>
#include "GTgraph/R-MAT/defs.h"
#include "GTgraph/R-MAT/init.h"
#include "GTgraph/R-MAT/graph.h"
}

// #define VALUETYPE double
// #define INDEXTYPE int
typedef double VALUETYPE;
typedef int INDEXTYPE;
// #define GEMM_DEBUG
#define HEAP_EXE
#define MKL_EXE
#define HASH_EXE
#define HASHVEC_EXE
#define ITERS 10

#ifdef CPP
#define MALLOC "new"
#elif defined IMM
#define MALLOC "mm"
#elif defined TBB
#define MALLOC "tbb"
#endif

enum generator_type
{
	rmat_graph,
	er_graph,
};


template <typename IT, typename NT>
bool sort_pair_less(const pair<IT, NT> &left,const pair<IT, NT> &right)
{
    return left.first < right.first;
}

template <typename IT>
bool sort_less_e(const IT &left, const IT &right)
{
    return left < right;
}

int main(int argc, char* argv[])
{
	bool binary = false;
	bool gen = false;
	string inputname1, inputname2, outputname;

	int edgefactor, scale, r_scale;
	generator_type gtype;
    vector<int> tnums;

	if (argc < 3) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale} {edgefactor} <numthreads>" << endl;
        return -1;
    }
    else if (argc < 6) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt|noout} <numthreads>" << endl;
#ifdef KNL_EXE
        cout << "Running on 68, 136, 204, 272 threads" << endl;
        tnums = {68, 136, 204, 272};
        // tnums = {1, 2, 4, 8, 16, 32, 64, 68, 128, 136, 192, 204, 256, 272}; // for scalability test
#else
        cout << "Running on 32, 64 threads" << endl; 
        tnums = {32, 64}; // for hashwell
#endif
    }
	else {
        cout << "Running on " << argv[5] << " processors" << endl;
        tnums = {atoi(argv[6])};
    }

	if (string(argv[1]) == string("gen")) {
        gen = true;
        cout << "Using synthetically generated data, with " << argv[2] << " generator of scale " << argv[3] << " and edgefactor " <<  argv[4] << endl;
        scale = atoi(argv[3]);
        edgefactor = atoi(argv[4]);
        r_scale = scale;
        if (string(argv[2]) == string("rmat")) {
            cout << "RMAT Generator" << endl;
            gtype = rmat_graph;
        }
        else {
            cout << "ER Generator" << endl;
            gtype = er_graph;
        }
    }
	else {
        inputname1 =  argv[2];
        string isbinary(argv[1]);
        if (isbinary == "text") {
            binary = false;
        }
        else if (isbinary == "binary") {
            binary = true;
        }
        else {
            cout << "unrecognized option, assuming text file" << endl;
        }
    }

	CSC<INDEXTYPE,VALUETYPE> * A_csc;
	if (gen) {
        double a, b, c, d;
        if(gtype == rmat_graph) {
            a = 0.57;
            b = 0.19;
            c = 0.19;
            d = 0.05;
        }
        else {
            a = b =  c = d = 0.25;
        }
        getParams();
        setGTgraphParams(scale, edgefactor, a, b, c, d);
        graph G1;
        graphGen(&G1);
        cerr << "Generator returned" << endl;
        A_csc = new CSC<INDEXTYPE,VALUETYPE> (G1);	// convert to CSC
        if (STORE_IN_MEMORY) {
            free(G1.start);
            free(G1.end);
            free(G1.w);
        }
    }
	else {
        if (binary) {
            ReadBinary( inputname1, *A_csc);
        }
        else {
            cout << "reading input matrices in text(ascii)... " << endl;
            ReadASCII( inputname1, *A_csc);
            stringstream ss1(inputname1);
            string cur;
            vector<string> v1;
            while (getline(ss1, cur, '.')) {
                v1.push_back(cur);
            }

            stringstream ss2(v1[v1.size() - 2]);
            vector<string> v2;
            while (getline(ss2, cur, '/')) {
                v2.push_back(cur);
            }
            inputname1 = v2[v2.size() - 1];
            
        }
    }
    
    /* Preprocess to A
     * Symmetrization
     * delete diagonal
     * Value = 1
     */
  	A_csc->Sorted();
    INDEXTYPE max_nnz = A_csc->nnz * 2;
    Triple<INDEXTYPE,VALUETYPE> * triples = new Triple<INDEXTYPE, VALUETYPE>[max_nnz];
    INDEXTYPE cnz = 0;
    for (INDEXTYPE i = 0; i < A_csc->cols; ++i) {
        for (INDEXTYPE j = A_csc->colptr[i]; j < A_csc->colptr[i + 1]; ++j) {
            INDEXTYPE rowid = A_csc->rowids[j];
            if (i != rowid) {
                triples[cnz].row = rowid;
                triples[cnz].col = i;
                triples[cnz].val = 1;
                cnz++;
                if (!(binary_search(&(A_csc->rowids[A_csc->colptr[rowid]]),
                                    &(A_csc->rowids[A_csc->colptr[rowid + 1]]),
                                    i))) {
                    triples[cnz].row = i;
                    triples[cnz].col = rowid;
                    triples[cnz].val = 1;
                    cnz++;
                }
            }
        }
    }
    
    cout << "nnz of symA: " << cnz << endl;
    
	CSC<INDEXTYPE, VALUETYPE> symA_csc(triples, cnz, A_csc->rows, A_csc->cols);
	CSC<INDEXTYPE, VALUETYPE> perm_symA_csc(cnz, A_csc->rows, A_csc->cols, 0);

    /* Permute row and column of symA_csc by degree */
    vector<pair<INDEXTYPE, INDEXTYPE>> nnz_perm(A_csc->cols);
    for (int i = 0; i < A_csc->cols; ++i) {
        nnz_perm[i].first = A_csc->colptr[i + 1] - A_csc->colptr[i];
        nnz_perm[i].second = i;
    }
    
    stable_sort(nnz_perm.begin(), nnz_perm.end(), sort_pair_less<INDEXTYPE, INDEXTYPE>);
    
    vector<INDEXTYPE> nnz_perm_read(A_csc->cols);
    for (int i = 0; i < A_csc->cols; ++i) {
        nnz_perm_read[nnz_perm[i].second] = i;
    }
    cnz = 0;
    for (int i = 0; i < symA_csc.cols; ++i) {
        perm_symA_csc.colptr[i] = cnz;
        int target_column = nnz_perm[i].second;
        for (int j = symA_csc.colptr[target_column]; j < symA_csc.colptr[target_column + 1]; ++j) {
            perm_symA_csc.rowids[cnz] = nnz_perm_read[symA_csc.rowids[j]];
            perm_symA_csc.values[cnz] = symA_csc.values[j];
            cnz++;
        }
    }
    
    perm_symA_csc.colptr[perm_symA_csc.cols] = cnz;
    
    cout << "nnz of perm_symA_csc: " << cnz << endl;
    
    symA_csc.make_empty();

    cout << "nnz of perm_symA: " << cnz << endl;
    
    INDEXTYPE L_nnz = perm_symA_csc.nnz / 2;
    
    CSC<INDEXTYPE, VALUETYPE> L_csc(L_nnz, perm_symA_csc.rows, perm_symA_csc.cols, 0);
    CSC<INDEXTYPE, VALUETYPE> U_csc(perm_symA_csc.nnz - L_nnz, perm_symA_csc.rows, perm_symA_csc.cols, 0);

    INDEXTYPE L_cnnz = 0, U_cnnz = 0;
    for (int i = 0; i < perm_symA_csc.cols; ++i) {
        L_csc.colptr[i] = L_cnnz;
        U_csc.colptr[i] = U_cnnz;
        for (int j = perm_symA_csc.colptr[i]; j < perm_symA_csc.colptr[i + 1]; ++j) {
            if (perm_symA_csc.rowids[j] > i) {
                L_csc.rowids[L_cnnz] = perm_symA_csc.rowids[j];
                L_csc.values[L_cnnz] = perm_symA_csc.values[j];
                L_cnnz++;
            }
            else {
                U_csc.rowids[U_cnnz] = perm_symA_csc.rowids[j];
                U_csc.values[U_cnnz] = perm_symA_csc.values[j];
                U_cnnz++;
            }
        }
    }
    L_csc.colptr[perm_symA_csc.cols] = L_cnnz;
    U_csc.colptr[perm_symA_csc.cols] = U_cnnz;
    
    // perm_symA_csc.make_empty();
    /* Sort perm_symA */
    for (INDEXTYPE i = 0; i < L_csc.cols; ++i) {
        stable_sort(&(L_csc.rowids[L_csc.colptr[i]]),
                    &(L_csc.rowids[L_csc.colptr[i + 1]]),
                    sort_less_e<INDEXTYPE>);
    }
    for (INDEXTYPE i = 0; i < U_csc.cols; ++i) {
        stable_sort(&(U_csc.rowids[U_csc.colptr[i]]),
                    &(U_csc.rowids[U_csc.colptr[i + 1]]),
                    sort_less_e<INDEXTYPE>);
    }
    
    cout << L_cnnz << ", " << U_cnnz << endl;
    

    /**
     ** Evaluation of SpGEMM between L and U_cnnz
     **/
    long long int nfop = get_flop(L_csc, U_csc);
    cout << "INTERMEDIATE PRODUCTS COUNT: " << nfop / 2 << endl;
    
    INDEXTYPE tri_count_l, tri_count_u, tri_count;
    double start, end, msec, ave_msec, mflops;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);

        /**
         ** Evaluate Heap SpGEMM
         **/
#ifdef HEAP_EXE
        CSC<INDEXTYPE,VALUETYPE> B_csc;
        HeapSpGEMM_lmalloc(L_csc, U_csc, B_csc, multiplies<VALUETYPE>(), plus<VALUETYPE>()); //first execution
        B_csc.make_empty();
        ave_msec = 0;
        for (int i = 0; i< ITERS; ++i) {
            start = omp_get_wtime();
            HeapSpGEMM_lmalloc(L_csc, U_csc, B_csc, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                B_csc.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;
    
        cout << "HeapSpGEMM returned with " << B_csc.nnz << " nonzeros" << endl;

        if (string(argv[1]) == string("gen")) {
            printf("sorted, heap_dcscmultcsc, %s, %d, %d, %d, %d, %f, %f\n", argv[2], tnum, scale, edgefactor, r_scale, ave_msec, mflops);
        }
        else {
            printf("sorted, heap_dcscmultcsc, %d, %s, %f, %f\n", tnum, inputname1.c_str(), ave_msec, mflops);
        }

        /* Mask operation to B by L */
        tri_count_l = 0;
        for (INDEXTYPE i = 0; i < L_csc.cols; ++i) {
            for (INDEXTYPE j = L_csc.colptr[i]; j < L_csc.colptr[i + 1]; ++j) {
                if ((binary_search(&(B_csc.rowids[B_csc.colptr[i]]),
                                    &(B_csc.rowids[B_csc.colptr[i + 1]]),
                                    L_csc.rowids[j]))) {
                    tri_count_l++;
                }
            }
        }

        tri_count_u = 0;
        for (INDEXTYPE i = 0; i < U_csc.rows; ++i) {
            for (INDEXTYPE j = U_csc.colptr[i]; j < U_csc.colptr[i + 1]; ++j) {
                if ((binary_search(&(B_csc.rowids[B_csc.colptr[i]]),
                                    &(B_csc.rowids[B_csc.colptr[i + 1]]),
                                    U_csc.rowids[j]))) {
                    tri_count_u++;
                }
            }
        }
        
        tri_count = tri_count_l + tri_count_u;
        cout << "Triangle count : " << tri_count_l << ", " << tri_count_u << ", " << tri_count << ", " << tri_count / 2 << endl;

        B_csc.make_empty();

#endif
    
#if defined(HASH_EXE) || defined(MKL_EXE) || defined(HASHVEC_EXE)
        /* Triangle counting does not allow unsorted output due to following mask operation */
        const bool sortOutput = true;
        string sortOutputStr = (sortOutput == true)? "sorted" : "unsorted";

        CSR<INDEXTYPE,VALUETYPE> L_csr (L_csc);    // convert to csr
        CSR<INDEXTYPE,VALUETYPE> U_csr (U_csc);    // convert to csr

        L_csr.Sorted();
        U_csr.Sorted();    
#endif
    
        /**
         ** Evaluate MKL Library
         **/
#ifdef MKL_EXE
        if(!L_csr.ConvertOneBased()) cout << "L was already one-based indexed" << endl;
        if(!U_csr.ConvertOneBased()) cout << "U was already one-based indexed" << endl;
    
        CSR<INDEXTYPE,VALUETYPE> B_csr_mkl;
        MKL_SpGEMM<sortOutput>(L_csr, U_csr, B_csr_mkl);
        B_csr_mkl.make_empty();
        ave_msec = 0;
    
        for (int i = 0; i< ITERS; ++i) {
            start = omp_get_wtime();
            MKL_SpGEMM<sortOutput>(L_csr, U_csr, B_csr_mkl);
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                B_csr_mkl.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;
        if (string(argv[1]) == string("gen")) {
            printf("%s, mkl_dcsrmultcsr, %s, %d, %d, %d, %d, %f, %f\n", sortOutputStr.c_str(), argv[2], tnum, scale, edgefactor, r_scale, ave_msec, mflops);
        }
        else {
            printf("%s, mkl_dcsrmultcsr, %d, %s, %f, %f\n", sortOutputStr.c_str(), tnum, inputname1.c_str(), ave_msec, mflops);
        }
        cout << "MKL returns " << B_csr_mkl.nnz << " non-zeros" << endl;

        /* Mask operation to B by L */
        tri_count_l = 0;
        for (INDEXTYPE i = 0; i < L_csr.rows; ++i) {
            for (INDEXTYPE j = L_csr.rowptr[i] - 1; j < L_csr.rowptr[i + 1] - 1; ++j) {
                if ((binary_search(&(B_csr_mkl.colids[B_csr_mkl.rowptr[i] - 1]),
                                    &(B_csr_mkl.colids[B_csr_mkl.rowptr[i + 1] - 1]),
                                    L_csr.colids[j]))) {
                    tri_count_l++;
                }
            }
        }

        tri_count_u = 0;
        for (INDEXTYPE i = 0; i < U_csr.rows; ++i) {
            for (INDEXTYPE j = U_csr.rowptr[i] - 1; j < U_csr.rowptr[i + 1] - 1; ++j) {
                if ((binary_search(&(B_csr_mkl.colids[B_csr_mkl.rowptr[i] - 1]),
                                    &(B_csr_mkl.colids[B_csr_mkl.rowptr[i + 1] - 1]),
                                    U_csr.colids[j]))) {
                    tri_count_u++;
                }
            }
        }
        
        tri_count = tri_count_l + tri_count_u;
        cout << "Triangle count : " << tri_count_l << ", " << tri_count_u << ", " << tri_count << ", " << tri_count / 2 << endl;

        B_csr_mkl.make_empty();
#endif

        /*
         * Evaluate HashSpGEMM
         */
#ifdef HASH_EXE
        L_csr.Sorted();
        U_csr.Sorted();    
    
        if(L_csr.ConvertZeroBased()) cout << "L was already zero-based indexed" << endl;
        if(U_csr.ConvertZeroBased()) cout << "U was already zero-based indexed" << endl;

        CSR<INDEXTYPE,VALUETYPE> B_csr;
        HashSpGEMM<false, sortOutput>(L_csr, U_csr, B_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        B_csr.make_empty();
    
        ave_msec = 0;
        for (int i = 0; i< ITERS; ++i) {
            start = omp_get_wtime();
            HashSpGEMM<false, sortOutput>(L_csr, U_csr, B_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                B_csr.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;
    
        cout << "HashSpGEMM returned with " << B_csr.nnz << " nonzeros" << endl;
        if (string(argv[1]) == string("gen")) {
            printf("%s, hash_dcsrmultcsr, %s, %d, %d, %d, %d, %f, %f\n", sortOutputStr.c_str(), argv[2], tnum, scale, edgefactor, r_scale, ave_msec, mflops);
        }
        else {
            printf("%s, hash_dcsrmultcsr, %d, %s, %f, %f\n", sortOutputStr.c_str(), tnum, inputname1.c_str(), ave_msec, mflops);
        }

        B_csr.Sorted();

        /* Mask operation to B by L */
        tri_count_l = 0;
        for (INDEXTYPE i = 0; i < L_csr.rows; ++i) {
            for (INDEXTYPE j = L_csr.rowptr[i]; j < L_csr.rowptr[i + 1]; ++j) {
                if ((binary_search(&(B_csr.colids[B_csr.rowptr[i]]),
                                    &(B_csr.colids[B_csr.rowptr[i + 1]]),
                                    L_csr.colids[j]))) {
                    tri_count_l++;
                }
            }
        }

        tri_count_u = 0;
        for (INDEXTYPE i = 0; i < U_csr.rows; ++i) {
            for (INDEXTYPE j = U_csr.rowptr[i]; j < U_csr.rowptr[i + 1]; ++j) {
                if ((binary_search(&(B_csr.colids[B_csr.rowptr[i]]),
                                    &(B_csr.colids[B_csr.rowptr[i + 1]]),
                                    U_csr.colids[j]))) {
                    tri_count_u++;
                }
            }
        }
        
        tri_count = tri_count_l + tri_count_u;
        cout << "Triangle count : " << tri_count_l << ", " << tri_count_u << ", " << tri_count << ", " << tri_count / 2 << endl;
        B_csr.make_empty();

#endif    

        /*
         * Evaluate HashVecSpGEMM
         */
#ifdef HASHVEC_EXE
        L_csr.Sorted();
        U_csr.Sorted();    
    
        if(L_csr.ConvertZeroBased()) cout << "L was already zero-based indexed" << endl;
        if(U_csr.ConvertZeroBased()) cout << "U was already zero-based indexed" << endl;

        CSR<INDEXTYPE,VALUETYPE> B_csr_vec;
        HashSpGEMM<true, sortOutput>(L_csr, U_csr, B_csr_vec, multiplies<VALUETYPE>(), plus<VALUETYPE>());
        B_csr_vec.make_empty();
    
        ave_msec = 0;
        for (int i = 0; i< ITERS; ++i) {
            start = omp_get_wtime();
            HashSpGEMM<true, sortOutput>(L_csr, U_csr, B_csr_vec, multiplies<VALUETYPE>(), plus<VALUETYPE>());
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                B_csr_vec.make_empty();
            }
        }
        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;
    
        cout << "HashSpGEMM returned with " << B_csr_vec.nnz << " nonzeros" << endl;
    
        if (string(argv[1]) == string("gen")) {
            printf("%s, hash_vec_dcsrmultcsr, %s, %d, %d, %d, %d, %f, %f\n", sortOutputStr.c_str(), argv[2], tnum, scale, edgefactor, r_scale, ave_msec, mflops);
        }
        else {
            printf("%s, hash_vec_dcsrmultcsr, %d, %s, %f, %f\n", sortOutputStr.c_str(), tnum, inputname1.c_str(), ave_msec, mflops);
        }

        B_csr_vec.Sorted();

        /* Mask operation to B by L */
        tri_count_l = 0;
        for (INDEXTYPE i = 0; i < L_csr.rows; ++i) {
            for (INDEXTYPE j = L_csr.rowptr[i]; j < L_csr.rowptr[i + 1]; ++j) {
                if ((binary_search(&(B_csr_vec.colids[B_csr_vec.rowptr[i]]),
                                    &(B_csr_vec.colids[B_csr_vec.rowptr[i + 1]]),
                                    L_csr.colids[j]))) {
                    tri_count_l++;
                }
            }
        }

        tri_count_u = 0;
        for (INDEXTYPE i = 0; i < U_csr.rows; ++i) {
            for (INDEXTYPE j = U_csr.rowptr[i]; j < U_csr.rowptr[i + 1]; ++j) {
                if ((binary_search(&(B_csr_vec.colids[B_csr_vec.rowptr[i]]),
                                    &(B_csr_vec.colids[B_csr_vec.rowptr[i + 1]]),
                                    U_csr.colids[j]))) {
                    tri_count_u++;
                }
            }
        }
        
        tri_count = tri_count_l + tri_count_u;
        cout << "Triangle count : " << tri_count_l << ", " << tri_count_u << ", " << tri_count << ", " << tri_count / 2 << endl;

        B_csr_vec.make_empty();

#endif    
    }
}
