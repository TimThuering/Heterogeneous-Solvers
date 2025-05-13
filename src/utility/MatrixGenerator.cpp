#include "MatrixGenerator.hpp"

#include "Configuration.hpp"

SymmetricMatrix MatrixGenerator::generateSPDMatrixStrictDiagonalDominant(sycl::queue& queue) {
    SymmetricMatrix matrix(conf::N, conf::matrixBlockSize, queue);

    // block count of all columns except the first one
    const int referenceBlockCount = (matrix.blockCountXY * (matrix.blockCountXY - 1)) / 2;

    // random number generator
    std::random_device rd;
    std::mt19937 generator(123);
    std::uniform_real_distribution<> distribution(-1.0, 1.0);

    for (std::size_t block_i = 0; block_i < matrix.blockCountXY; ++block_i) {
        for (std::size_t block_j = 0; block_j <= block_i; ++block_j) {
            // number of blocks in row to the right (if matrix would be full)
            const int block_j_inv = matrix.blockCountXY - (block_j + 1);

            // total number of blocks to the right that are stored
            const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

            // id of block in the matrix data structure for symmetric matrices
            const int blockID = block_i + referenceBlockCount - columnBlocksToRight;

            // start index of block in matrix data structure
            const int blockStartIndex = blockID * conf::matrixBlockSize * conf::matrixBlockSize;


            if (block_i == block_j) {
                // Diagonal block
                for (std::size_t i = 0; i < matrix.blockSize; ++i) {
                    for (std::size_t j = 0; j <= i; ++j) {
                        if (block_i * conf::matrixBlockSize + i < conf::N &&
                            block_j * conf::matrixBlockSize + j < conf::N) {
                            const conf::fp_type value = distribution(generator);
                            if (i == j) {
                                matrix.matrixData[blockStartIndex + i * conf::matrixBlockSize + j] += std::abs(value);
                            } else {
                                // location (i,j)
                                matrix.matrixData[blockStartIndex + i * conf::matrixBlockSize + j] = value;
                                // mirrored value in upper triangle (j,i)
                                matrix.matrixData[blockStartIndex + j * conf::matrixBlockSize + i] = value;
                            }
                        }
                    }
                }
            } else {
                // Non-diagonal block
                for (std::size_t i = 0; i < matrix.blockSize; ++i) {
                    for (std::size_t j = 0; j < matrix.blockSize; ++j) {
                        if (block_i * conf::matrixBlockSize + i < conf::N &&
                            block_j * conf::matrixBlockSize + j < conf::N) {
                            const conf::fp_type value = distribution(generator);
                            matrix.matrixData[blockStartIndex + i * conf::matrixBlockSize + j] = value;
                        }
                    }
                }
            }
        }
    }


    return matrix;
}

RightHandSide MatrixGenerator::generateRHS(sycl::queue& queue) {
    RightHandSide b(conf::N, conf::matrixBlockSize, queue);

    // random number generator
    std::random_device rd;
    std::mt19937 generator(321);
    std::uniform_real_distribution<> distribution(-1.0, 1.0);

    for (std::size_t i = 0; i < conf::N; ++i) {
        const conf::fp_type value = distribution(generator);
        b.rightHandSideData[i] = value;
    }
    return b;
}
