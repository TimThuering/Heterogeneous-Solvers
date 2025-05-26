#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "Configuration.hpp"
#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"
#include "MatrixOperations.hpp"

using namespace sycl;
class TriangularSolveTest : public ::testing::Test {
protected:
    std::string path_A = "../tests/testData/testMatrixSymmetric20x20.txt";
};

TEST_F(TriangularSolveTest, triangularSolveTest) {
}
