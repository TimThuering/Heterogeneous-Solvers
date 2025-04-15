#ifndef VECTOROPERATIONS_HPP
#define VECTOROPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"


class VectorOperations {
public:
    /**
     * Scales (part of) a vector with a scalar in parallel
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param x vector x to be scaled
     * @param alpha scalar to scale the vector
     * @param result result vector
     * @param blockStart_i first block in the vector to apply the scaling to
     * @param blockCount_i amount of blocks after first block to be scaled
     */
    static void scaleVectorBlock(sycl::queue& queue, const conf::fp_type* x, conf::fp_type alpha, conf::fp_type* result,
                                 int blockStart_i, int blockCount_i);

    /**
     * Adds (parts of) two vectors x + y in parallel
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param x vector x
     * @param y vector y
     * @param result result vector
     * @param blockStart_i first block on which the addition should be performed
     * @param blockCount_i amount of blocks from after first block to be added
     */
    static void addVectorBlock(sycl::queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                               conf::fp_type* result,
                               int blockStart_i, int blockCount_i);

    /**
     * Subtracts (parts of) two vectors x + y in parallel
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param x vector x
     * @param y vector y
     * @param result result vector
     * @param blockStart_i first block on which the subtraction should be performed
     * @param blockCount_i amount of blocks from after first block to be subtracted
     */
    static void subVectorBlock(sycl::queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                               conf::fp_type* result,
                               int blockStart_i, int blockCount_i);

    /**
     * Computes the first part of the scalar product x*y. The result are partial sums of each work-group that have to be
     * summed up to obtain the final result.
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param x vector x
     * @param y vector y
     * @param result vector that stores the partial sums of each work-group
     * @param blockStart_i start block of the vector
     * @param blockCount_i block count of the vector
     */
    static void scalarProduct(sycl::queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                              conf::fp_type* result,
                              int blockStart_i, int blockCount_i);

    /**
     * Sums up the partial results from the first part of the scalar product computation to obtain the final scalar product.
     * The result will be stored in result[0]
     *
     * @param queue SYCL queue that determines the device for the parallel execution
     * @param result vector that stores the partial sums of each work-group
     * @param workGroupCount work-group count from the previous step
     */
    static void sumFinalScalarProduct(sycl::queue& queue, conf::fp_type* result, int workGroupCount);
};


#endif //VECTOROPERATIONS_HPP
