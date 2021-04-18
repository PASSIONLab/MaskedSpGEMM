cmake_minimum_required(VERSION 3.14)
project(MaskedSpGEMM
        VERSION 0.1
        LANGUAGES C CXX)

macro(add_example _name _src)
    add_executable(${_name} ${_src}
            GTgraph/R-MAT/graph.o
            GTgraph/R-MAT/utils.o
            GTgraph/R-MAT/init.o
            GTgraph/R-MAT/globals.o
            PaRMAT/Release/src/GraphGen_notSorted.o
            PaRMAT/Release/src/utils.o
            PaRMAT/Release/src/Edge.o
            PaRMAT/Release/src/Square.o
            PaRMAT/Release/src/GraphGen_sorted.o)

    target_include_directories(${_name} PRIVATE GTgraph/sprng2.0-lite/include)
    target_link_directories(${_name} PRIVATE GTgraph/sprng2.0-lite/lib)
    target_link_libraries(${_name} libsprng.a)

    find_package(OpenMP REQUIRED)
    target_link_libraries(${_name} OpenMP::OpenMP_CXX)

#    target_include_directories(${_name} PRIVATE /opt/tbb/include/)
#    target_link_libraries(${_name} /opt/tbb/lib/intel64/gcc4.8/libtbb.so)
#    target_link_libraries(${_name} /opt/tbb/lib/intel64/gcc4.8/libtbbmalloc.so)

    target_compile_definitions(${_name} PRIVATE CPP)

    target_compile_features(${_name} PUBLIC cxx_std_17)
endmacro()

add_example(masked-all sample/MaskedAllSpGEMM.cpp)
add_example(tricnt-all sample/TriCntAll.cpp)

add_subdirectory(test)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "C compiler flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    message(STATUS "C compiler flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    message(STATUS "C compiler flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else ()
    message("Build type unknown.")
endif ()
