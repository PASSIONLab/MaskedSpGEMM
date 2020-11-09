#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <random>
#include <thread>

#include "../CSC.h"
#include "../CSR.h"
#include "../IO.h"
#include "../utility.h"
#include "../PaRMAT/src/GraphGen_sorted.hpp"
#include "../PaRMAT/src/GraphGen_notSorted.hpp"
#include "../PaRMAT/src/utils.hpp"
#include "../PaRMAT/src/internal_config.hpp"

using namespace std;

extern "C" {
#include "../GTgraph/R-MAT/defs.h"
#include "../GTgraph/R-MAT/graph.h"
#include "../GTgraph/R-MAT/init.h"
}

#ifndef _SAMPLE_COMMON_HPP_
#define _SAMPLE_COMMON_HPP_

#ifdef CPP
#define MALLOC "new"
#elif defined IMM
#define MALLOC "mm"
#elif defined TBB
#define MALLOC "tbb"
#else
#define MALLOC "tbb"
#endif


enum generator_type {
  rmat_graph,
  er_graph,
};

template <class INDEXTYPE, class VALUETYPE>
void GenRMAT_Par(CSC<INDEXTYPE, VALUETYPE> &A_csc, int64_t nVertices, int64_t nEdges, double a, double b, double c)
{
    double RAM_usage = 0.5;
    unsigned long long standardCapacity = 0;



    // try to manage their numbers automatically. If cannot determine, go single-threaded.
    unsigned int nCPUWorkerThreads = std::max( 1, static_cast<int>(std::thread::hardware_concurrency()) - 1 );

    // Avoiding very small regions which may cause incorrect results.
    if( nEdges < 10000 )
        nCPUWorkerThreads = 1;

    auto totalSystemRAM = static_cast<unsigned long long>(getTotalSystemMemory());   // In bytes.
    auto availableSystemRAM = calculateAvailableRAM( totalSystemRAM, RAM_usage );    // In bytes.

    standardCapacity = availableSystemRAM / (2*nCPUWorkerThreads*sizeof(Edge)); // 2 can count for vector's effect.
    std::cout << "Each thread capacity is " << standardCapacity << " edges." << "\n";

    double start = omp_get_wtime();

    // GenerateGraph generates vertices 0...n range
    // hence, we passed nVertices-1
    // danger: do not directedGraph to false (then multithreaded setting gets to infiniteloop in utils.cpp generate_edges function line 156)
    std::vector<std::pair<int64_t, int64_t>> edges =
    GraphGen_notSorted::GenerateGraph(
                                      nEdges, nVertices-1,
                                      a, b, c,
                                      nCPUWorkerThreads,
                                      standardCapacity,
                                      true, //allowEdgeToSelf,
                                      false, //allowDuplicateEdges,
                                      true //directedGraph
                                      );

    double end = omp_get_wtime();
    cerr << "Generator time: " << end-start << " seconds"<< endl;
    A_csc = CSC<INDEXTYPE, VALUETYPE>(edges, nEdges, nVertices, nVertices);
}

template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSC(CSC<INDEXTYPE, VALUETYPE> &A_csc,
                           CSC<INDEXTYPE, VALUETYPE> &B_csc, char **argv) {
  bool binary = false;
  bool gen = false;
  string inputname1, inputname2, outputname;
  int edgefactor, scale, r_scale;
  generator_type gtype;

  A_csc.make_empty();
  B_csc.make_empty();

  if (string(argv[1]) == string("gen")) {
    gen = true;
    cout << "Using synthetically generated data of scale " << argv[3]
         << " and edgefactor " << argv[4] << endl;
    scale = atoi(argv[3]);
    edgefactor = atoi(argv[4]);
    r_scale = scale;
    if (string(argv[2]) == string("rmat")) {
      cout << "RMAT Generator" << endl << endl;
      gtype = rmat_graph;
    } else {
      // Erdos-Renyi graph
      cout << "ER Generator" << endl << endl;
      gtype = er_graph;
    }
  } else {
    inputname1 = argv[2];
    inputname2 = argv[3];
    string isbinary(argv[1]);
    if (isbinary == "text") {
      binary = false;
    } else if (isbinary == "binary") {
      binary = true;
    } else {
      cout << "unrecognized option, assuming text file" << endl;
    }
  }

  if (gen) {
    double a, b, c, d;
    if (gtype == rmat_graph) {
      a = 0.57;
      b = 0.19;
      c = 0.19;
      d = 0.05;
    } else {
      a = b = c = d = 0.25;
    }

    getParams();
    setGTgraphParams(scale, edgefactor, a, b, c, d);
    graph G1;
    graphGen(&G1);
    cerr << "Generator returned" << endl;
    A_csc = *(new CSC<INDEXTYPE, VALUETYPE>(G1));
    if (STORE_IN_MEMORY) {
      free(G1.start);
      free(G1.end);
      free(G1.w);
    }
    graph G2;
    graphGen(&G2);
    cerr << "Generator returned" << endl;
    B_csc = *(new CSC<INDEXTYPE, VALUETYPE>(G2));

    if (STORE_IN_MEMORY) {
      free(G2.start);
      free(G2.end);
      free(G2.w);
    }

      // // generate using parallel generator
      // int64_t nVertices = pow(2, scale);
      // int64_t nEdges = nVertices * edgefactor;;
      // GenRMAT_Par(A_csc, nVertices, nEdges, a, b, c);
      // cerr << "Generator returned for the first matrix" << endl;
      // GenRMAT_Par(B_csc, nVertices, nEdges, a, b, c);
      // cerr << "Generator returned for the second matrix" << endl;
  } else {
    if (binary) {
      ReadBinary(inputname1, A_csc);
      ReadBinary(inputname2, B_csc);
    } else {
      cout << "reading input matrices in text (ascii)... " << endl;
      ReadASCII(inputname1, A_csc);
      ReadASCII(inputname2, B_csc);
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
}

template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSR(CSR<INDEXTYPE, VALUETYPE> &A_csr,
                           CSR<INDEXTYPE, VALUETYPE> &B_csr, char **argv) {
  CSC<INDEXTYPE, VALUETYPE> A_csc, B_csc;

  A_csr.make_empty();
  B_csr.make_empty();

  SetInputMatricesAsCSC(A_csc, B_csc, argv);

  A_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));
  B_csr = *(new CSR<INDEXTYPE, VALUETYPE>(B_csc));
}

#endif
