#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "Configuration.hpp"
#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"

using namespace sycl;

class MatrixOperationsTest : public ::testing::Test {
protected:
    std::string path_A = "../tests/testData/testMatrixSymmetric20x20.txt";
};

TEST_F(MatrixOperationsTest, fullMatrixVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728,
        -1.086646607850169, 0.066961689956242, 0.108626371550612, -0.193678763374773,
        0.390679188452535, -1.356673858842381, 0.127116701062358, -0.09906986830946,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674,
        -0.505032158088103, -0.05679402522748, -0.630967426449773, -0.033817780168102
    };
}
