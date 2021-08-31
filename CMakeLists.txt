cmake_minimum_required(VERSION 3.20)
project(CuspCluster CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CuspCluster serialtest2.cu)

add_compile_options(-gencode arch=compute_50, code=compute_50)

ADD_DEFINITIONS(-DDEBUG)

set_target_properties(CuspCluster PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
