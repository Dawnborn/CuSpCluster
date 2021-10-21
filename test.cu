//
// Created by hjp on 2021/9/12.
//

#include <gtest/gtest.h>
#include <iostream>
#include "serialtest2.h"

std::string bpath = "./../../data/blob/";
std::string jpath = "./../../data/jester_dataset/";
std::string tpath = "./../../data/unittest_dataset/"; //path for unit test

TEST(MyTest, CLusterUpdate1){
    int a = 1;
    int b = 1;
    EXPECT_EQ(a,1);
    unittest_ClusterUpdate_MembershipUpdate<double,int>(tpath,MoreParallelReduction);
}

TEST(MyTest, MembershipUpdate1){
    int b = 1;
    EXPECT_EQ(b,1);
}

int main(int argc, char *argv[]){
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}