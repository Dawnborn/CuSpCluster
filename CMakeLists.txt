cmake_minimum_required(VERSION 3.20)
project(CuspCluster CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

find_package(glog REQUIRED)

add_executable(CuspCluster serialtest2.cu)

add_compile_options(-gencode arch=compute_50, code=compute_50)

ADD_DEFINITIONS(-DDEBUG)

set_target_properties(CuspCluster PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CuspCluster gflags)
target_link_libraries(CuspCluster glog::glog)
