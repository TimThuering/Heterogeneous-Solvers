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

// Block size = 4 --> no padding

// vector scale

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

// vector add

TEST_F(vectorOperationsTest, addFullVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // add complete vectors b + b
    MatrixOperations::addVectorBlock(queue, b.rightHandSideData.data(), b.rightHandSideData.data(), result.data(), 0,
                                     b.blockCountX);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.047286498801027, 1.801854785303741, -1.423361549121465, 1.794597788548975,
        -0.752674191958058, -0.306694204109697, 1.310810375281767, -0.363203454523355,
        0.198374750692238, -1.889763547027727, 1.014052434699226, 0.152573252877113,
        -0.681073134003631, 1.153714813713617, -0.78722068283342, -0.186008442077394,
        -1.463833211011341, -0.387548054211483, -1.186179037295402, -0.950746638232602
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(vectorOperationsTest, addUpperVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // add upper 3 blocks of b + b
    MatrixOperations::addVectorBlock(queue, b.rightHandSideData.data(), b.rightHandSideData.data(), result.data(), 0,
                                     3);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.047286498801027, 1.801854785303741, -1.423361549121465, 1.794597788548975,
        -0.752674191958058, -0.306694204109697, 1.310810375281767, -0.363203454523355,
        0.198374750692238, -1.889763547027727, 1.014052434699226, 0.152573252877113,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(vectorOperationsTest, addLowerVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // add lower 2 blocks of b + b
    MatrixOperations::addVectorBlock(queue, b.rightHandSideData.data(), b.rightHandSideData.data(), result.data(), 3,
                                     2);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        -0.681073134003631, 1.153714813713617, -0.78722068283342, -0.186008442077394,
        -1.463833211011341, -0.387548054211483, -1.186179037295402, -0.950746638232602
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

// vector sub

TEST_F(vectorOperationsTest, subFullVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);

    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> y(allocator);
    y.resize(b.rightHandSideData.size());

    // initialize y with some data
    for (unsigned int i = 0; i < y.size(); ++i)
    {
        y[i] = std::sqrt(i);
    }

    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // add complete vectors b - y
    MatrixOperations::subVectorBlock(queue, b.rightHandSideData.data(), y.data(), result.data(), 0,
                                     b.blockCountX);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.023643249400513, -0.099072607348129, -2.125894336933828, -0.834751913294389,
        -2.376337095979029, -2.389415079554638, -1.794084555142294, -2.827353038326268,
        -2.729239749400072, -3.944881773513863, -2.655251442818766, -3.240338163916843,
        -3.80463818213957, -3.02869386860718, -4.135267728190652, -3.965987567246114,
        -4.731916605505671, -4.316879652723403, -4.835730205766986, -4.834272262656975
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(vectorOperationsTest, subUpperVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);

    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> y(allocator);
    y.resize(b.rightHandSideData.size());

    // initialize y with some data
    for (unsigned int i = 0; i < y.size(); ++i)
    {
        y[i] = std::sqrt(i);
    }


    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // add upper 3 blocks of b - y
    MatrixOperations::subVectorBlock(queue, b.rightHandSideData.data(), y.data(), result.data(), 0,
                                     3);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.023643249400513, -0.099072607348129, -2.125894336933828, -0.834751913294389,
        -2.376337095979029, -2.389415079554638, -1.794084555142294, -2.827353038326268,
        -2.729239749400072, -3.944881773513863, -2.655251442818766, -3.240338163916843,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(vectorOperationsTest, subLowerVector)
{
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);

    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> y(allocator);
    y.resize(b.rightHandSideData.size());

    // initialize y with some data
    for (unsigned int i = 0; i < y.size(); ++i)
    {
        y[i] = std::sqrt(i);
    }


    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // sub lower 2 blocks of b - y
    MatrixOperations::subVectorBlock(queue, b.rightHandSideData.data(), y.data(), result.data(), 3,
                                     2);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        -3.80463818213957, -3.02869386860718, -4.135267728190652, -3.965987567246114,
        -4.731916605505671, -4.316879652723403, -4.835730205766986, -4.834272262656975
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}


