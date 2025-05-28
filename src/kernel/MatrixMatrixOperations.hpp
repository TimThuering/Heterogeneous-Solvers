#ifndef MATRIXMATRIXOPERATIONS_HPP
#define MATRIXMATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"

class MatrixMatrixOperations {
public:
    /**
     * This function solves a triangular System LB=B for a (sub) column of the matrix A.
     * L is a lower triangular matrix and corresponds to a diagonal block of A, after the cholesky decomposition has been
     * performed on this block.
     * B can be any sequence of blocks below the diagonal block, starting at row specified through block start and ending
     * blockCount blocks below.
     *
     * The method launches a kernel on the device asynchronously and returns after that.
     * One has to wait to ensure correctness.
     *
     * @param queue sycl queue for the device where the code will execute
     * @param A complete matrix A, on which the decomposition is performed
     * @param blockID id of the diagonal block that holds the triangular matrix of the system to be solved
     * @param blockRow row of the block with the triangular system
     * @param blockStart row, in which the first system will be solved
     * @param blockCount specifies, for how many blocks below blockStart the system will be solved
     * @return a sycl event of the kernel execution
     */
    static sycl::event triangularSolve(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow, int blockStart,
                                       int blockCount);

    /**
     * This function solves a triangular System LB=B for a (sub) column of the matrix A.
     * L is a lower triangular matrix and corresponds to a diagonal block of A, after the cholesky decomposition has been
     * performed on this block.
     * B can be any sequence of blocks below the diagonal block, starting at row specified through block start and ending
     * blockCount blocks below.
     *
     * The kernel launched by this function is optimized for execution on GPUs.
     *
     * The method launches a kernel on the device asynchronously and returns after that.
     * One has to wait to ensure correctness.
     *
     * @param queue sycl queue for the device where the code will execute
     * @param A complete matrix A, on which the decomposition is performed
     * @param blockID id of the diagonal block that holds the triangular matrix of the system to be solved
     * @param blockRow row of the block with the triangular system
     * @param blockStart row, in which the first system will be solved
     * @param blockCount specifies, for how many blocks below blockStart the system will be solved
     * @return a sycl event of the kernel execution
     */
    static sycl::event triangularSolve_optimizedGPU(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow,
                                                 int blockStart, int blockCount);

    static sycl::event symmetricMatrixMatrixDiagonal(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow,
                                             int blockStart, int blockCount, int  blockCountXY);
};


#endif //MATRIXMATRIXOPERATIONS_HPP
