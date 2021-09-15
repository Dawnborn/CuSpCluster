cmake_minimum_required(VERSION 3.20)
project(CuspCluster CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})
find_package(glog REQUIRED)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
find_package(Threads REQUIRED)

add_executable(CuspCluster main.cu serialtest2.h)
add_executable(MyTests test.cu)

add_compile_options(-gencode arch=compute_50, code=compute_50)
add_definitions(-DDEBUG)

set_target_properties(CuspCluster PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(MyTests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CuspCluster gflags)
target_link_libraries(CuspCluster glog::glog)

target_link_libraries(MyTests ${GTEST_BOTH_LIBRARIES} pthread)
target_link_libraries(MyTests ${CMAKE_THREADS_LIBS_INIT})
