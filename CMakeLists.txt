cmake_minimum_required(VERSION 3.20)
project(CuspCluster CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CuspCluster serialtest.cu)

add_compile_options(-gencode arch=compute_50,code=compute_50)

set_target_properties(CuspCluster PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
