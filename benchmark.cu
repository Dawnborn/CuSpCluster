/** generate executable for nvvp **/

#include "serialtest2.h"
#include <iostream>

int main() {
    std::string bpath = "./../../data/blob/";
    std::string jpath = "./../../data/jester_dataset/";
    std::string tpath = "./../../data/unittest_dataset/"; //path for unit test
    cudaDeviceReset();
//    CallfuncSync<double, int, 101, 10, 24938, 1>(jpath, MoreParallelReduction,dst1);
//    CallfuncSync<double, int, 101, 10, 24938, 1>(jpath, MoreParallelReduction,dst2);
//    CallfuncSync<double,int,10,6,30,100>(bpath, MoreParallelReduction,dst1);
//    CallfuncSync<double,int,10,6,30,100>(bpath, SharedMemory,dst1);
//    CallfuncSync<double, int, 101, 200, 24938, 50>(jpath, SharedMemory,dst1);

//    CallfuncSync<double, int, 128, 100, 24938, 50>(jpath, SharedMemory,dst4);
    CallfuncSync<double, int, 128, 200, 24938, 50>(jpath, SharedMemory,dst31);
//    cudaDeviceReset();
//    CallfuncSync<double, int, 128, 200, 24938, 20>(jpath, SharedMemory,dst31);
}