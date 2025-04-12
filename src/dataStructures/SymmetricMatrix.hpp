#ifndef SYMMETRICMATRIX_HPP
#define SYMMETRICMATRIX_HPP

#include <vector>
#include <sycl/sycl.hpp>

#include "Configuration.hpp"

/**
 * Class that represents a symmetric matrix that is stored in a blocked manner.
 * Symmetric values in diagonal blocks are stored redundantly. Otherwise, only the lower triangle of the matrix is stored.
 *
 * The block size can be set via the constructor.
 *
 * Blocks in the lower (blocked) triangle of the matrix are enumerated from top to bottom and from left to right in the matrix.
 * Internally blocks are stored after each other according to their ID.
 * Each block itself is stored in row major layout.
 *
 * Example:
 *
 * Lower triangle of the symmetric matrix divided into blocks:
 * |---+---+---|
 * | 0 |   |   |
 * |---+---+---|
 * | 1 | 3 |   |
 * |---+---+---|
 * | 2 | 4 | 5 |
 * |---+---+---|
 *
 *
 * Block order in memory:
 *
 * | 0 | 1 | 2 | 3 | 4 | 5 |
 *
 */
class SymmetricMatrix {
public:
    /**
     * Constructor of the class.
     * Automatically resizes the vector matrixData to the correct size.
     */
    SymmetricMatrix(std::size_t N, int blockSize, sycl::queue &queue);

    const std::size_t N; /// Size N of the NxN symmetric matrix
    const int blockSize; /// The matrix will be partitioned in blockSize x blockSize blocks
    const int blockCountXY; /// block Count in X/Y direction (if the matrix would be stored completely)
    sycl::queue &cpuQueue;

    std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>> matrixData; /// internal matrix data structure allocated as SYCL host memory
};

#endif //SYMMETRICMATRIX_HPP
