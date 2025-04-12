#include "VectorOperations.hpp"

using namespace sycl;

void VectorOperations::scaleVectorBlock(queue &queue, const conf::fp_type *x, const conf::fp_type alpha,
                                        conf::fp_type *result, const int blockStart_i, const int blockCount_i) {
    // global range corresponds to number of rows in the (sub) vector
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    queue.submit([&](handler &h) {
        h.parallel_for(kernelRange, [=](auto &nd_item) {
            // row i in the matrix
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            result[i] = x[i] * alpha;
        });
    });
}

void VectorOperations::addVectorBlock(queue &queue, const conf::fp_type *x, const conf::fp_type *y,
                                      conf::fp_type *result,
                                      const int blockStart_i, const int blockCount_i) {
    // global range corresponds to number of rows in the (sub) vector
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    queue.submit([&](handler &h) {
        h.parallel_for(kernelRange, [=](auto &nd_item) {
            // row i in the matrix
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            result[i] = x[i] + y[i];
        });
    });
}

void VectorOperations::subVectorBlock(queue &queue, const conf::fp_type *x, const conf::fp_type *y,
                                      conf::fp_type *result, const int blockStart_i, const int blockCount_i) {
    // global range corresponds to number of rows in the (sub) vector
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    queue.submit([&](handler &h) {
        h.parallel_for(kernelRange, [=](auto &nd_item) {
            // row i in the matrix
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            result[i] = x[i] - y[i];
        });
    });
}
