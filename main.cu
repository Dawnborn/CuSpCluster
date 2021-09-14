//
// Created by hjp on 2021/9/12.
//

#include "serialtest2.h"
#include <iostream>

int main(){
    std::string path = "./../../data/blob/";
    std::string jpath = "./../../data/jester_dataset/";
//    CallfuncSync<double,int>(cpu,path);
//    CallfuncSync<double,int>(naive,path);
//    CallfuncSync<double,int>(SharedMemory,path); //FIXME: changed not changed? no changed?
//    CallfuncSync<double,int>(ParallelReduction,path);
    CallfuncSync<double,int>(MoreParallelReduction,path);

//    std::string tpath = "./../../data/unittest_dataset/"; //path for unit test
//    unittest_ClusterUpdate_MembershipUpdate<double,int>(tpath,MoreParallelReduction);
    return 0;
}
