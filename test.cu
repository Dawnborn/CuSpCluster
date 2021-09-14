//
// Created by hjp on 2021/9/12.
//

#include <gtest/gtest.h>
#include "serialtest2.h"

TEST(MyTest, CLusterUpdate1){
    int a = 1;
    int b = 1;
    EXPECT_EQ(a,1);
}

TEST(MyTest, MembershipUpdate1){
    int b = 1;
    EXPECT_EQ(b,1);
}

int main(int argc, char *argv[]){
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}