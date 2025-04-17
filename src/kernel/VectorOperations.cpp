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

unsigned int VectorOperations::scalarProduct(queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                                     conf::fp_type* result,
                                     const int blockStart_i, const int blockCount_i) {
    const unsigned int matrixBlockSize = conf::matrixBlockSize;
    const unsigned int workGroupSize = conf::workGroupSizeVector;
    const unsigned int vectorLength = blockCount_i * matrixBlockSize;

    assert((vectorLength) % 2 == 0);

    const unsigned int globalSize = vectorLength / 2;
    const unsigned int workGroupCount = std::ceil(static_cast<conf::fp_type>(globalSize) / static_cast<conf::fp_type>(workGroupSize));
    const unsigned int globalSizePadding = workGroupCount * workGroupSize;

    assert(globalSizePadding % workGroupSize == 0);

    // global range corresponds to half (!) of the number of rows in the (sub) vector
    // each work-item will perform the first add operation when loading data from global memory
    const range globalRange(globalSizePadding);
    const range localRange(workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};


    // based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    queue.submit([&](handler& h) {
        local_accessor<conf::fp_type> cache(workGroupSize, h);

        h.parallel_for(kernelRange, [=](nd_item<1>& nd_item) {
            // row i in the matrix
            const int offset = blockStart_i * matrixBlockSize;
            const unsigned int localID = nd_item.get_local_id();
            const unsigned int globalIndex = offset + nd_item.get_group(0) * (workGroupSize * 2) + localID;

            cache[localID] = 0;
            if (globalIndex < offset + vectorLength) {
                cache[localID] += x[globalIndex] * y[globalIndex];
            }
            if (globalIndex + workGroupSize < offset + vectorLength) {
                cache[localID] += x[globalIndex + workGroupSize] * y[globalIndex + workGroupSize];
            }
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

    return workGroupCount;
}

void VectorOperations::sumFinalScalarProduct(queue& queue, conf::fp_type* result, const unsigned int workGroupCount) {

    const unsigned int workGroupSize = conf::workGroupSizeFinalScalarProduct;

    if (2 * workGroupSize < workGroupCount) {
        throw std::runtime_error("Workgroup size for final scalar product is too small");
    }

    // use maximum available global range
    const range globalRange(workGroupSize);
    const range localRange(workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};


    // based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    queue.submit([&](handler& h) {
        local_accessor<conf::fp_type> cache(workGroupSize, h);

        h.parallel_for(kernelRange, [=](nd_item<1>& nd_item) {
            // row i in the matrix
            const unsigned int localID = nd_item.get_local_id();
            const unsigned int globalIndex = localID;

            cache[localID] = 0;
            if (globalIndex < workGroupCount) {
                cache[localID] += result[globalIndex];
            }
            if (globalIndex + workGroupSize < workGroupCount) {
                cache[localID] += result[globalIndex + workGroupSize];
            }
            nd_item.barrier();

            for (unsigned int stride =  workGroupSize / 2; stride > 0; stride = stride / 2) {
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
