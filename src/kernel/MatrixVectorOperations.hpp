#ifndef MATRIXVECTOROPERATIONS_HPP
#define MATRIXVECTOROPERATIONS_HPP

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
     * @return a sycl event of the kernel execution
     */
    static sycl::event matrixVectorBlock(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result, int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY, bool reset = true);

    /**
     * Parallel SYCL implementation of a blocked matrix vector product Ab = x on the symmetric matrix data structure.
     * Depending on the arguments the complete matrix vector product is calculated or only a sub part of the matrix and the vector are multiplied with each other.
     *
     * This kernel is optimized for the execution on GPUs by using local memory.
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
     * @return a sycl event of the kernel execution
     */
    static sycl::event matrixVectorBlock_GPU(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result, int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY, bool reset = true);

    /**
     * Parallel SYCL implementation of a blocked matrix vector product Ab = x on the symmetric matrix data structure.
     * Depending on the arguments the complete matrix vector product is calculated or only a sub part of the matrix and the vector are multiplied with each other.
     *
     * This kernel enables AVX on the CPU which might not yield better performance in all scenarios.
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
     * @return a sycl event of the kernel execution
     */
    static sycl::event matrixVectorBlock_CPU(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result, int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY, bool reset = true);

    /**
     * Parallel SYCL implementation of a triangular solve for a single matrix block and one (sub)-vector.
     * Used to as one step in the blocking algorithm to solve the system of equations with a lower triangular matrix produced by the cholesky decomposition.
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param A the lower triangular matrix
     * @param b the right hand side
     * @param blockRow the row in which the block is located that should be used to solve the system
     * @param blockID ID of the block that should be used to solve the system
     * @param transposed true or false, if the matrix A should be interpreted as transposed, i.e., an upper triangular matrix
     * @return a sycl event of the kernel execution
     */
    static sycl::event triangularSolveBlockVector(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b, int blockRow, int blockID, bool transposed);

    /**
     * Parallel SYCL implementation of the column update step for a blocked triangular solve algorithm.
     * Updates either the whole column or the upper or lower part of it.
     * If the matrix is interpreted as transposed, i.e., an upper triangular matrix, "upper" means the left part of the row in the lower triangular, non-transposed, matrix.
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param A the lower triangular matrix
     * @param b the right hand side
     * @param blockStart offset, how many blocks to start below the diagonal (non-transposed) or how many blocks to start from the top/left (transposed)
     * @param blockCount how many blocks below (non-transposed) or right (transposed) of the first block should be updated
     * @param blockRow the row of the current diagonal block. if transposed, the corresponding row in the lower-triangular, non-transposed matrix has to be specified
     * @param blockID ID of the block that should be used to solve the system
     * @param blockCountXY block Count in X/Y direction of the matrix A
     * @param transposed true or false, if the matrix A should be interpreted as transposed, i.e., an upper triangular matrix
     * @return a sycl event of the kernel execution
     */
    static sycl::event matrixVectorColumnUpdate(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b, int blockStart, int blockCount, int blockRow, int blockID, int blockCountXY, bool transposed);


    /**
     * Performs a naive, unoptimized classical matrix-vector product that gets used in the Gaussian Process pipeline.
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param A matrix A
     * @param b right hand side b
     * @param result result of A times b
     * @param n row count of A
     * @param m column count of A
     * @return a sycl event of the kernel execution
     */
    static sycl::event matrixVectorGP(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b, conf::fp_type* result, int n, int m);
};


#endif //MATRIXVECTOROPERATIONS_HPP
