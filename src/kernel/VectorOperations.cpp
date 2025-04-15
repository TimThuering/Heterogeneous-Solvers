#include "VectorOperations.hpp"

using namespace sycl;

void VectorOperations::scaleVectorBlock(queue& queue, const conf::fp_type* x, const conf::fp_type alpha,
                                        conf::fp_type* result, const int blockStart_i, const int blockCount_i) {
    // global range corresponds to number of rows in the (sub) vector
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    queue.submit([&](handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            // row i in the vector
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            result[i] = x[i] * alpha;
        });
    });
}

void VectorOperations::addVectorBlock(queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                                      conf::fp_type* result,
                                      const int blockStart_i, const int blockCount_i) {
    // global range corresponds to number of rows in the (sub) vector
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    queue.submit([&](handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            // row i in the vector
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            result[i] = x[i] + y[i];
        });
    });
}

void VectorOperations::subVectorBlock(queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                                      conf::fp_type* result, const int blockStart_i, const int blockCount_i) {
    // global range corresponds to number of rows in the (sub) vector
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    queue.submit([&](handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            // row i in the vector
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            result[i] = x[i] - y[i];
        });
    });
}

void VectorOperations::scalarProduct(queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                                     conf::fp_type* result,
                                     const int blockStart_i, const int blockCount_i) {
    const int matrixBlockSize = conf::matrixBlockSize;
    const int workGroupSize = conf::workGroupSizeVector;

    // global range corresponds to half (!) of the number of rows in the (sub) vector
    // each work-item will perform the first add operation when loading data from global memory
    const range globalRange(blockCount_i * matrixBlockSize / 2);
    const range localRange(workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};


    // based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    queue.submit([&](handler& h) {
        local_accessor<conf::fp_type> cache(workGroupSize, h);

        h.parallel_for(kernelRange, [=](nd_item<1>& nd_item) {
            // row i in the matrix
            const int offset = blockStart_i * matrixBlockSize;
            const unsigned localID = nd_item.get_local_id();
            const unsigned int globalIndex = offset + nd_item.get_group(0) * (workGroupSize * 2) + localID;

            cache[localID] = x[globalIndex] * y[globalIndex] + x[globalIndex + workGroupSize] * y[globalIndex +
                workGroupSize];

            nd_item.barrier();

            for (unsigned int stride = workGroupSize / 2; stride > 0; stride = stride / 2) {
                if (localID < stride) {
                    cache[localID] += cache[localID + stride];
                }
                nd_item.barrier();
            }

            if (localID == 0) {
                result[nd_item.get_group(0)] = cache[0];
            }
        });
    });
}

void VectorOperations::sumFinalScalarProduct(queue& queue, conf::fp_type* result, const int workGroupCount) {

    int workGroupSize = conf::workGroupSizeVector;

    if (workGroupSize < workGroupCount) {
        workGroupSize = workGroupCount;
    }

    // global range corresponds to half (!) of the number of rows in the (sub) vector
    // each work-item will perform the first add operation when loading data from global memory
    const range globalRange(workGroupSize / 2);
    const range localRange(workGroupSize / 2);
    const auto kernelRange = nd_range{globalRange, localRange};


    // based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    queue.submit([&](handler& h) {
        local_accessor<conf::fp_type> cache(workGroupSize, h);

        h.parallel_for(kernelRange, [=](nd_item<1>& nd_item) {
            // row i in the matrix
            const unsigned localID = nd_item.get_local_id();
            const unsigned int globalIndex = localID;

            cache[localID] = 0;
            if (globalIndex < workGroupCount) {
                cache[localID] += result[globalIndex];
            }
            if (globalIndex + nd_item.get_local_range(0) < workGroupCount) {
                cache[localID] += result[globalIndex + nd_item.get_local_range(0)];
            }
            nd_item.barrier();

            for (unsigned int stride =  nd_item.get_local_range(0) / 2; stride > 0; stride = stride / 2) {
                if (localID < stride) {
                    cache[localID] += cache[localID + stride];
                }
                nd_item.barrier();
            }

            if (localID == 0) {
                result[nd_item.get_group(0)] = cache[0];
            }
        });
    });
}
