#ifndef MATRIXOPERATIONS_HPP
#define MATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"


class MatrixVectorOperations {
public:
    /**
     * Parallel SYCL implementation of a blocked matrix vector product Ab = x on the symmetric matrix data structure.
     * Depending on the arguments the complete matrix vector product is calculated or only a sub part of the matrix and the vector are multiplied with each other.
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param A the symmetric matrix A
     * @param b vector b
     * @param result result vector
     * @param blockStart_i row index of first block of sub-matrix
     * @param blockStart_j column index of first block of sub-matrix
     * @param blockCount_i block count in row direction of sub-matrix
     * @param blockCount_j block count in column direction of sub-matrix
     * @param blockCountXY block count in x and y direction of the complete symmetric matrix
     * @param reset if true (default), the existing entries in the result vector will be ignored. If false, the result will be added to the values in the result vector.
     */
    static sycl::event matrixVectorBlock(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result, int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY, bool reset = true);

    static sycl::event matrixVectorBlock_GPU(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result, int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY, bool reset = true);

    static sycl::event matrixVectorBlock_CPU(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result, int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY, bool reset = true);

    static sycl::event triangularSolveVector(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b, int blockStart, int blockCount, int blockRow, int blockID, bool transposed);
};


#endif //MATRIXOPERATIONS_HPP
