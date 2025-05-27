#ifndef MATRIXOPERATIONS_HPP
#define MATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"


class MatrixOperations {
public:
    /**
     * This method performs a cholesky decomposition on a diagonal block of the matrix A.
     *
     * The method launches a kernel on the device asynchronously and returns after that.
     * One has to wait to ensure correctness.
     *
     * @param queue sycl queue of the device the kernel should be executed on
     * @param A complete matrix, on which the overall blocked Cholesky is performed
     * @param blockID ID of the diagonal block on which this method should perform a decomposition
     * @param blockRow row of the current diagonal block that is processed
     * @return a sycl event of the kernel execution
     */
    static sycl::event cholesky(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow);

    /**
     * This method performs a cholesky decomposition on a diagonal block of the matrix A.
     * The kernel launched by this method is optimized for the execution on GPUs.
     * It makes use of local memory.
     *
     * The method launches a kernel on the device asynchronously and returns after that.
     * One has to wait to ensure correctness.
     *
     * @param queue sycl queue of the device the kernel should be executed on
     * @param A complete matrix, on which the overall blocked Cholesky is performed
     * @param blockID ID of the diagonal block on which this method should perform a decomposition
     * @param blockRow row of the current diagonal block that is processed
     * @return a sycl event of the kernel execution
     */
    static sycl::event cholesky_GPU(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow);

    /**
     * This method performs a cholesky decomposition on a diagonal block of the matrix A.
     * The kernel launched by this method is optimized for the execution on GPUs.
     * It uses local memory and additional work-items to set the upper diagonal to zero.
     *
     * The method launches a kernel on the device asynchronously and returns after that.
     * One has to wait to ensure correctness.
     *
     * @param queue sycl queue of the device the kernel should be executed on
     * @param A complete matrix, on which the overall blocked Cholesky is performed
     * @param blockID ID of the diagonal block on which this method should perform a decomposition
     * @param blockRow row of the current diagonal block that is processed
     * @return a sycl event of the kernel execution
     */
    static sycl::event cholesky_GPU_optimized(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow);
};


#endif //MATRIXOPERATIONS_HPP
