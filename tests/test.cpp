#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "matrixData/SymmetricMatrix.hpp"

TEST(Test, test1)
{
    SymmetricMatrix matrix;
    int result = matrix.example();

    EXPECT_EQ(result, 123);
}