#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "MatrixOperations.hpp"
#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"
#include "RightHandSide.hpp"
#include "Configuration.hpp"

using namespace sycl;

class vectorOperationsTest : public ::testing::Test
{
protected:
    std::string path_A = "../tests/testData/testMatrixSymmetric20x20.txt";
    std::string path_b = "../tests/testData/testVector_20.txt";
};

TEST_F(vectorOperationsTest, scaleFullVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    MatrixOperations::scaleVectorBlock(queue, b.rightHandSideData.data(), 1.23456, result.data(), 0, b.blockCountX);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.029189009979898, 1.112248921872294, -0.878612617041698, 1.107769322915512,
        -0.46461072521187, -0.189316198312834, 0.809137028453929, -0.224198228408177,
        0.122452766107305, -1.166513242309275, 0.625954286891139, 0.094180417535984,
        -0.420412824157762, 0.712165080209142, -0.485935583099414, -0.114819291125534,
        -0.903594964493081, -0.239225662903664, -0.732204596141706, -0.586876884848221
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(vectorOperationsTest, scaleUpperVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // scale upper 3 blocks of b
    MatrixOperations::scaleVectorBlock(queue, b.rightHandSideData.data(), 1.23456, result.data(), 0, 3);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.029189009979898, 1.112248921872294, -0.878612617041698, 1.107769322915512,
        -0.46461072521187, -0.189316198312834, 0.809137028453929, -0.224198228408177,
        0.122452766107305, -1.166513242309275, 0.625954286891139, 0.094180417535984,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(vectorOperationsTest, scaleLowerVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // scale lower 2 blocks of b
    MatrixOperations::scaleVectorBlock(queue, b.rightHandSideData.data(), 1.23456, result.data(), 3, 2);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        -0.420412824157762, 0.712165080209142, -0.485935583099414, -0.114819291125534,
        -0.903594964493081, -0.239225662903664, -0.732204596141706, -0.586876884848221
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}
