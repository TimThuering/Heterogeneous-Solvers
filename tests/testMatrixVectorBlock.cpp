#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "MatrixVectorOperations.hpp"
#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"
#include "RightHandSide.hpp"
#include "Configuration.hpp"

using namespace sycl;

class MatrixVectorTest : public ::testing::Test {
protected:
    std::string path_A = "../tests/testData/testMatrixSymmetric20x20.txt";
    std::string path_b = "../tests/testData/testVector_20.txt";
};


class TRSVTest : public ::testing::Test {
protected:
    std::string path_A = "../tests/testData/testMatrixSymmetric20x20.txt";


    std::vector<conf::fp_type> reference = {
        0.3603902513595023,
        0.4647412009353097,
        2.7253161533854984,
        0.3007735318732128,
        0.5872207510409728,
        1.5852712666315125,
        1.0594776221362836,
        0.388333972364927,
        2.236407028666025,
        0.6022134289757832,
        1.945987295528439,
        0.9270990706988526,
        3.849916171818029,
        0.4383131764729952,
        1.233587121224736,
        0.4072011951795052,
        1.4980895787658652,
        0.44939017504969897,
        0.6765944977364532,
        1.9716118473800197,
        0.,
        0.,
        0.,
        0.
    };

    std::vector<conf::fp_type> reference_transposed = {
        0.7561826008952903,
        0.43422165948281244,
        0.160310312151354,
        0.7533143653652702,
        -0.02937398483250901,
        1.6721678989524362,
        0.42113623510921244,
        1.8668199375622456,
        1.1535173486635255,
        -0.17712268119970628,
        1.7754034138387895,
        1.250993694774386,
        0.9901546147145652,
        1.3011398651252468,
        1.0343013726618364,
        -0.32476586630754983,
        2.2457110290152356,
        4.181373549733097,
        2.6264495546679623,
        1.6160004168501227,
        0.,
        0.,
        0.,
        0.
    };

    std::vector<conf::fp_type> reference_one_block = {
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        2.640689705963016,
        1.2854913706980609,
        0.9921551039564678,
        0.3499738886590734,
        2.0604900442948364,
        2.4782760557241335,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.
    };
};


// Block size 4 --> no padding

TEST_F(MatrixVectorTest, fullMatrixVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              A.blockCountXY, A.blockCountXY, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728,
        -1.086646607850169, 0.066961689956242, 0.108626371550612, -0.193678763374773,
        0.390679188452535, -1.356673858842381, 0.127116701062358, -0.09906986830946,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674,
        -0.505032158088103, -0.05679402522748, -0.630967426449773, -0.033817780168102
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, upperMatrixVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;

    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // upper 3 blocks of A times full b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              3, A.blockCountXY, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728,
        -1.086646607850169, 0.066961689956242, 0.108626371550612, -0.193678763374773,
        0.390679188452535, -1.356673858842381, 0.127116701062358, -0.09906986830946,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, lowerMatrixVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;

    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // lower 2 blocks of A times full b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              0,
                                              2, A.blockCountXY, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674,
        -0.505032158088103, -0.05679402522748, -0.630967426449773, -0.033817780168102
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}


TEST_F(MatrixVectorTest, topLeftMatrixTopVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;

    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // upper left 3x3 blocks of A times upper 3 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              3, 3, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        1.406146742004936, -1.601333413734391, -0.238543333079249, 2.212721971246077,
        -1.964346280441348, 0.201131008354004, -0.317434875011155, -0.52246006571351,
        0.761878496149202, -1.920733189257554, 0.143177380952106, -0.412051278909714,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, lowerRightMatrixBottomVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;

    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // lower right 2x2 blocks of A times lower 2 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              3,
                                              2, 2, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        -0.132743480888271, 0.264378169770815, -0.534263176501096, 0.674498274717188,
        -0.358037604274327, 0.088946472265775, -0.079525956151658, -0.285531853757102
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, upperRightMatrixBottomVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;

    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // upper right 3x2 blocks of A times lower 2 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              3,
                                              3, 2, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        -0.730972800479452, 0.844115897524262, -0.205058302914846, -0.639990401973277,
        0.877699672591179, -0.134169318397762, 0.426061246561767, 0.328781302338737,
        -0.371199307696667, 0.564059330415172, -0.016060679889748, 0.312981410600254,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, lowerLeftMatrixTopVector) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;

    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // lower left 2x3 blocks of A times upper 2 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              0,
                                              2, 3, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.110465790771175, -0.709557891968216, 0.985819174737127, -1.745704821830862,
        -0.146994553813777, -0.145740497493255, -0.551441470298115, 0.251714073589
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, fullMatrixVectorBlocked) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 4;
    conf::workGroupSize = 4;

    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // upper left 3x3 blocks of A times upper 3 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              3, 3, A.blockCountXY, true);

    // lower right 2x2 blocks of A times lower 2 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              3,
                                              2, 2, A.blockCountXY, true);

    queue.wait();

    // upper right 3x2 blocks of A times lower 2 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              3,
                                              3, 2, A.blockCountXY, false);


    // lower left 2x3 blocks of A times upper 2 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              0,
                                              2, 3, A.blockCountXY, false);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728,
        -1.086646607850169, 0.066961689956242, 0.108626371550612, -0.193678763374773,
        0.390679188452535, -1.356673858842381, 0.127116701062358, -0.09906986830946,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674,
        -0.505032158088103, -0.05679402522748, -0.630967426449773, -0.033817780168102
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}


// Block size 6 --> padding

TEST_F(MatrixVectorTest, fullMatrixVectorPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              A.blockCountXY, A.blockCountXY, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728, -1.086646607850169,
        0.066961689956242,
        0.108626371550612, -0.193678763374773, 0.390679188452535, -1.356673858842381, 0.127116701062358,
        -0.09906986830946,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674, -0.505032158088103,
        -0.05679402522748,
        -0.630967426449773, -0.033817780168102, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, upperMatrixVectorPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // upper 3 blocks of A times full b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              3, A.blockCountXY, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728, -1.086646607850169,
        0.066961689956242,
        0.108626371550612, -0.193678763374773, 0.390679188452535, -1.356673858842381, 0.127116701062358,
        -0.09906986830946,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674, -0.505032158088103,
        -0.05679402522748,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, lowerMatrixVectorPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // lower 2 blocks of A times full b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              0,
                                              1, A.blockCountXY, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -0.630967426449773, -0.033817780168102, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, topLeftMatrixTopVectorPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // upper left 3x3 blocks of A times upper 3 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              3, 3, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.585572327946026, -0.592101300186843, -0.372720082547384, 1.430925106740686, -0.966487866525573,
        0.027828763500268,
        0.105195363016181, -0.209796518413056, 0.375713291311535, -1.231613857011912, 0.079802783578683,
        -0.022591572560161,
        0.005615911612721, -0.34098197332125, 0.366465774353217, -0.850549426681102, -0.496908130122445,
        -0.109629888732084,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}


TEST_F(MatrixVectorTest, lowerRightMatrixBottomVectorPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // lower right 1x1 block of A times lower 1 block of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              3,
                                              1, 1, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -0.218503589679951, -0.254913243503526, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, upperRightMatrixBottomVectorPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // upper right 3x1 blocks of A times lower 1 block of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              3,
                                              3, 1, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.089601613579457, -0.165116216023286, -0.070881553446711, 0.141806462532114, -0.120158741324596,
        0.039132926455975,
        0.003431008534431, 0.016117755038283, 0.014965897141, -0.12506000183047, 0.047313917483675, -0.076478295749299,
        -0.027893601729818, -0.104197748876151, 0.085090223882815, -0.220657120432571, -0.008124027965658,
        0.052835863504604,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, lowerLeftMatrixTopVectorPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    // lower left 1x3 blocks of A times upper 1 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              0,
                                              1, 3, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -0.412463836769823, 0.221095463335423, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}

TEST_F(MatrixVectorTest, fullMatrixVectorBlockedPadding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 3;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());

    // upper left 3x3 blocks of A times upper 3 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              0,
                                              3, 3, A.blockCountXY, true);

    // lower right 1x1 blocks of A times lower 1 block of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              3,
                                              1, 1, A.blockCountXY, true);

    queue.wait();

    // upper right 3x1 blocks of A times lower 1 block of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                              3,
                                              3, 1, A.blockCountXY, false);


    // lower left 1x3 blocks of A times upper 3 blocks of b vector
    MatrixVectorOperations::matrixVectorBlock(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 3,
                                              0,
                                              1, 3, A.blockCountXY, false);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728, -1.086646607850169,
        0.066961689956242,
        0.108626371550612, -0.193678763374773, 0.390679188452535, -1.356673858842381, 0.127116701062358,
        -0.09906986830946,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674, -0.505032158088103,
        -0.05679402522748,
        -0.630967426449773, -0.033817780168102, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}


TEST_F(MatrixVectorTest, fullMatrixVectorPadding_SharedMemory) {
    queue queue(gpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 6;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);
    RightHandSide b = MatrixParser::parseRightHandSide(path_b, queue);


    const usm_allocator<conf::fp_type, usm::alloc::shared> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::shared>> result(allocator);
    result.resize(b.rightHandSideData.size());


    MatrixVectorOperations::matrixVectorBlock_GPU(queue, A.matrixData.data(), b.rightHandSideData.data(), result.data(), 0,
                                                  0,
                                                  A.blockCountXY, A.blockCountXY, A.blockCountXY);
    queue.wait();

    std::vector<conf::fp_type> reference = {
        0.675173941525483, -0.757217516210128, -0.443601635994095, 1.5727315692728, -1.086646607850169,
        0.066961689956242,
        0.108626371550612, -0.193678763374773, 0.390679188452535, -1.356673858842381, 0.127116701062358,
        -0.09906986830946,
        -0.022277690117097, -0.445179722197401, 0.451555998236032, -1.071206547113674, -0.505032158088103,
        -0.05679402522748,
        -0.630967426449773, -0.033817780168102, 0.0, 0.0, 0.0, 0.0
    };

    EXPECT_EQ(result.size(), reference.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], reference[i], 1e-12);
    }
}


// Tests for TRSV kernel
TEST_F(TRSVTest, testTRSV) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 20;
    conf::workGroupSize = 20;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);


    const usm_allocator<conf::fp_type, usm::alloc::shared> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::shared>> b(allocator);
    b.resize(20);
    for (size_t i = 0; i < 20; i++) {
        b[i] = 1;
    }


    MatrixVectorOperations::triangularSolveVector(queue, A.matrixData.data(), b.data(), 0, 1, 0, 0, false);
    queue.wait();


    for (size_t i = 0; i < b.size(); i++) {
        EXPECT_NEAR(b[i], reference[i], 1e-12);
    }
}

TEST_F(TRSVTest, testTRSV_transposed) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 20;
    conf::workGroupSize = 20;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);


    const usm_allocator<conf::fp_type, usm::alloc::shared> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::shared>> b(allocator);
    b.resize(20);
    for (size_t i = 0; i < 20; i++) {
        b[i] = 1;
    }


    MatrixVectorOperations::triangularSolveVector(queue, A.matrixData.data(), b.data(), 0, 1, 0, 0, true);
    queue.wait();


    for (size_t i = 0; i < b.size(); i++) {
        EXPECT_NEAR(b[i], reference_transposed[i], 1e-12);
    }
}

TEST_F(TRSVTest, testTRSV_padding) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 24;
    conf::workGroupSize = 24;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);


    const usm_allocator<conf::fp_type, usm::alloc::shared> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::shared>> b(allocator);
    b.resize(24);
    for (size_t i = 0; i < 20; i++) {
        b[i] = 1;
    }


    MatrixVectorOperations::triangularSolveVector(queue, A.matrixData.data(), b.data(), 0, 1, 0, 0, false);
    queue.wait();


    for (size_t i = 0; i < b.size(); i++) {
        EXPECT_NEAR(b[i], reference[i], 1e-12);
    }
}

TEST_F(TRSVTest, testTRSV_padding_transposed) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 24;
    conf::workGroupSize = 24;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);


    const usm_allocator<conf::fp_type, usm::alloc::shared> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::shared>> b(allocator);
    b.resize(24);
    for (size_t i = 0; i < 20; i++) {
        b[i] = 1;
    }


    MatrixVectorOperations::triangularSolveVector(queue, A.matrixData.data(), b.data(), 0, 1, 0, 0, true);
    queue.wait();


    for (size_t i = 0; i < b.size(); i++) {
        EXPECT_NEAR(b[i], reference_transposed[i], 1e-12);
    }
}

TEST_F(TRSVTest, testTRSV_one_block) {
    queue queue(cpu_selector_v);
    conf::matrixBlockSize = 6;
    conf::workGroupSize = 6;
    SymmetricMatrix A = MatrixParser::parseSymmetricMatrix(path_A, queue);


    const usm_allocator<conf::fp_type, usm::alloc::shared> allocator{queue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::shared>> b(allocator);
    b.resize(24);
    for (size_t i = 12; i < 18; i++) {
        b[i] = 1;
    }


    MatrixVectorOperations::triangularSolveVector(queue, A.matrixData.data(), b.data(), 0, 1, 2, 7, false);
    queue.wait();


    for (size_t i = 0; i < b.size(); i++) {
        EXPECT_NEAR(b[i], reference_one_block[i], 1e-12);
    }
}
