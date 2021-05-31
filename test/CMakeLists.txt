macro(add_test_exe _name)
    add_executable(test-${_name} ${_name}-test.cpp)
    target_compile_definitions(test-${_name} PRIVATE CPP)
    target_compile_features(test-${_name} PUBLIC cxx_std_17)


    find_package(OpenMP REQUIRED)
    target_link_libraries(test-${_name} OpenMP::OpenMP_CXX)
endmacro()

add_test_exe(scan)
add_test_exe(hash-accumulator)
add_test_exe(sparse-accumulator)
add_test_exe(spgemm)
add_test_exe(split-mem)
add_test_exe(memalign)
add_test_exe(triple-util)