cmake_minimum_required(VERSION 3.14)
project(MaskedSpGEMM
        VERSION 0.1
        LANGUAGES C CXX)

macro(add_example _name _src)
    message(${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(${_name} ${_src}
            ${PROJECT_SOURCE_DIR}/GTgraph/R-MAT/graph.o
            ${PROJECT_SOURCE_DIR}/GTgraph/R-MAT/utils.o
            ${PROJECT_SOURCE_DIR}/GTgraph/R-MAT/init.o
            ${PROJECT_SOURCE_DIR}/GTgraph/R-MAT/globals.o
            ${PROJECT_SOURCE_DIR}/PaRMAT/Release/src/GraphGen_notSorted.o
            ${PROJECT_SOURCE_DIR}/PaRMAT/Release/src/utils.o
            ${PROJECT_SOURCE_DIR}/PaRMAT/Release/src/Edge.o
            ${PROJECT_SOURCE_DIR}/PaRMAT/Release/src/Square.o
            ${PROJECT_SOURCE_DIR}/PaRMAT/Release/src/GraphGen_sorted.o)

    target_include_directories(${_name} PRIVATE ${PROJECT_SOURCE_DIR}/GTgraph/sprng2.0-lite/include)
    target_link_directories(${_name} PRIVATE ${PROJECT_SOURCE_DIR}/GTgraph/sprng2.0-lite/lib)
    target_link_libraries(${_name} libsprng.a)

	target_include_directories(${_name} PRIVATE ${PROJECT_SOURCE_DIR}/GraphBLAS/Include)
	target_link_directories(${_name} PRIVATE ${PROJECT_SOURCE_DIR}/GraphBLAS/build)
	target_link_libraries(${_name} libgraphblas.so)

    find_package(OpenMP REQUIRED)
    target_link_libraries(${_name} OpenMP::OpenMP_CXX)

    #    target_include_directories(${_name} PRIVATE /opt/tbb/include/)
    #    target_link_libraries(${_name} /opt/tbb/lib/intel64/gcc4.8/libtbb.so)
    #    target_link_libraries(${_name} /opt/tbb/lib/intel64/gcc4.8/libtbbmalloc.so)

    target_compile_definitions(${_name} PRIVATE CPP)

    target_compile_features(${_name} PUBLIC cxx_std_17)
endmacro()

include(FetchContent)
FetchContent_Declare(
        otx
        GIT_REPOSITORY https://github.com/srki/otx.git
)
FetchContent_MakeAvailable(otx)

add_example(masked-all benchmarks/MaskedAllSpGEMM.cpp)
add_example(complement-masked-all benchmarks/ComplementMaskedSpGEMM.cpp)
add_example(tricnt-all benchmarks/TriCntAll.cpp)
add_example(heatmap benchmarks/HeatMap.cpp)
add_example(tricnt-all-grb benchmarks/TriCntAllGrb.cpp)
add_example(ktruss-all-grb benchmarks/KTrussAllGrb.cpp)
# target_link_libraries(heatmap otx-cpp)
# add_example(test-grb sample/test_grb.cpp)

if (BUILD_TESTS)
add_subdirectory(test)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "C compiler flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "C compiler flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "C compiler flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else ()
    message("Build type unknown.")
endif ()
