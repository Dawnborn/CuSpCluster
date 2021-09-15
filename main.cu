//
// Created by hjp on 2021/9/12.
//

#include "serialtest2.h"
#include <iostream>

int main() {
    std::string bpath = "./../../data/blob/";
    std::string jpath = "./../../data/jester_dataset/";
    std::string tpath = "./../../data/unittest_dataset/"; //path for unit test
//    unittest_ClusterUpdate_MembershipUpdate<double,int>(tpath,MoreParallelReduction);

//    CallfuncSync<double,int>(cpu,path);
//    CallfuncSync<double,int>(naive,path);
//    CallfuncSync<double,int>(SharedMemory,path); //FIXME: changed not changed? no changed?
//    CallfuncSync<double,int>(ParallelReduction,path);
//    CallfuncSync<double,int,10,6,30,100>(MoreParallelReduction,bpath);
    CallfuncSync<double, int, 101, 10, 24938, 200>(MoreParallelReduction, jpath);

//    CallfuncSingleStream<double,int>(naive,path);

    return 0;
}
