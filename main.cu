//
// Created by hjp on 2021/9/12.
//

#include "serialtest2.h"
#include <iostream>

int main() {
    std::string bpath = "./../../data/blob/";
    std::string jpath = "./../../data/jester_dataset/";
    std::string tpath = "./../../data/unittest_dataset/"; //path for unit test

//    CallfuncSync<double,int,10,6,30,100>(MoreParallelReduction,bpath);
//    CallfuncSync<double,int,10,6,30,100>(bpath, SharedMemory,dst1);
//    CallfuncSync<double, int, 101, 100, 24938, 50>(jpath, SharedMemory,dst1);
//    CallfuncSync<double,int,10,6,30,100>(bpath,SharedMemory,dst2);
    CallfuncSync<double,int,16,3,30,100>(tpath,SharedMemory2,dst1);

    return 0;
}
