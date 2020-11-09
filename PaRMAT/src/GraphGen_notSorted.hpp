#ifndef GRAPH_GEN_NOT_SORTED_HPP
#define GRAPH_GEN_NOT_SORTED_HPP

#include <fstream>
#include <vector>

namespace GraphGen_notSorted{

	bool GenerateGraph(
			const unsigned long long nEdges,
			const unsigned long long nVertices,
			const double a, const double b, const double c,
			const unsigned int nCPUWorkerThreads,
			std::ofstream& outFile,
			const unsigned long long standardCapacity,
			const bool allowEdgeToSelf,
			const bool allowDuplicateEdges,
			const bool directedGraph
			);
    std::vector<std::pair<int64_t, int64_t>>
    GenerateGraph(
                  const unsigned long long nEdges,
                  const unsigned long long nVertices,
                  const double RMAT_a, const double RMAT_b, const double RMAT_c,
                  const unsigned int nCPUWorkerThreads,
                  const unsigned long long standardCapacity,
                  const bool allowEdgeToSelf,
                  const bool allowDuplicateEdges,
                  const bool directedGraph
                  );

};

#endif	//	GRAPH_GEN_NOT_SORTED_HPP
