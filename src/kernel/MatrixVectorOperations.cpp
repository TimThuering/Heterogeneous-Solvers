#include <sycl/sycl.hpp>

#include "MatrixVectorOperations.hpp"
#include "SymmetricMatrix.hpp"

using namespace sycl;

sycl::event MatrixVectorOperations::matrixVectorBlock(queue& queue, const conf::fp_type* A, const conf::fp_type* b,
                                                      conf::fp_type* result,
                                                      const int blockStart_i,
                                                      const int blockStart_j, const int blockCount_i,
                                                      const int blockCount_j,
                                                      const int blockCountXY, const bool reset) {
    // global range corresponds to number of rows in the (sub) matrix
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    const bool addToPreviousEntries = !reset;


    sycl::event event = queue.submit([&](handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            // row i in the matrix
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            // local row index in current matrix block
            const int iInBlock = i % matrixBlockSize;

            // row index of matrix block current work-item will work with
            const int block_i = sycl::floor(static_cast<double>(i) / matrixBlockSize);

            // block count of all columns except the first one
            const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

            // block ID in the symmetric matrix
            int blockID = 0;

            // first index for block columns
            int block_j = blockStart_j;

            conf::fp_type resultValue = 0;
            if (addToPreviousEntries) {
                resultValue += result[i];
            }

            // First step: Process all matrix blocks up to the diagonal block (included) or the most left block that should be processed
            // the blocks can be interpreted as they are stored in memory
            for (; block_j <= min(block_i, blockStart_j + blockCount_j - 1); ++block_j) {
                // number of blocks in row to the right (if matrix would be full)
                const int block_j_inv = blockCountXY - (block_j + 1);

                // total number of blocks to the right that are stored
                const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_i + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;
                int rowStartIndex = blockStartIndex + iInBlock * matrixBlockSize;

                // go through all columns of the block and compute the matrix vector product
                for (int j = 0; j < matrixBlockSize; ++j) {
                    resultValue += A[rowStartIndex + j] * b[block_j * matrixBlockSize + j];
                }
            }


            // Second step: Process all matrix blocks after the diagonal block
            // the blocks have to be interpreted as transposed, since the upper triangle is not stored explicitly
            for (; block_j < blockStart_j + blockCount_j; ++block_j) {
                // same block ID calculation as previously, but now block_i and block_j have to be swapped due to symmetries
                const int block_i_inv = blockCountXY - (block_i + 1);
                const int columnBlocksToRight = (block_i_inv * (block_i_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_j + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                const int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;

                // go through all columns of the block and compute the matrix vector product
                // the block in storage now has to be interpreted as transposed since we are working on the data of the symmetric block
                for (int j = 0; j < matrixBlockSize; ++j) {
                    resultValue += A[blockStartIndex + j * matrixBlockSize + iInBlock] * b[block_j * matrixBlockSize +
                        j];
                }
            }

            // store the result
            result[i] = resultValue;
        });
    });

    return event;
}

sycl::event MatrixVectorOperations::matrixVectorBlock_GPU(sycl::queue& queue, const conf::fp_type* A,
                                                          const conf::fp_type* b,
                                                          conf::fp_type* result, const int blockStart_i,
                                                          const int blockStart_j, const int blockCount_i,
                                                          const int blockCount_j, const int blockCountXY,
                                                          const bool reset) {
    // global range corresponds to number of rows in the (sub) matrix
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    const bool addToPreviousEntries = !reset;


    sycl::event event = queue.submit([&](handler& h) {
        auto local_b = local_accessor<conf::fp_type, 1>(conf::workGroupSize, h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            // row i in the matrix
            const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

            // local row index in current matrix block
            const int iInBlock = i % matrixBlockSize;

            // row index of matrix block current work-item will work with
            const int block_i = sycl::floor(static_cast<double>(i) / matrixBlockSize);

            // block count of all columns except the first one
            const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

            // block ID in the symmetric matrix
            int blockID = 0;

            // first index for block columns
            int block_j = blockStart_j;

            conf::fp_type resultValue = 0;
            if (addToPreviousEntries) {
                resultValue += result[i];
            }

            // First step: Process all matrix blocks up to the diagonal block (included) or the most left block that should be processed
            // the blocks can be interpreted as they are stored in memory
            for (; block_j <= min(block_i, blockStart_j + blockCount_j - 1); ++block_j) {
                // number of blocks in row to the right (if matrix would be full)
                const int block_j_inv = blockCountXY - (block_j + 1);

                // total number of blocks to the right that are stored
                const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_i + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;
                int rowStartIndex = blockStartIndex + iInBlock * matrixBlockSize;

                // cache part of rhs b in local memory
                nd_item.barrier();
                local_b[nd_item.get_local_id()] = b[block_j * matrixBlockSize + nd_item.get_local_id()];
                nd_item.barrier();

                // go through all columns of the block and compute the matrix vector product
                for (int j = 0; j < matrixBlockSize; ++j) {
                    resultValue += A[rowStartIndex + j] * local_b[j];
                }
            }


            // Second step: Process all matrix blocks after the diagonal block
            // the blocks have to be interpreted as transposed, since the upper triangle is not stored explicitly
            for (; block_j < blockStart_j + blockCount_j; ++block_j) {
                // same block ID calculation as previously, but now block_i and block_j have to be swapped due to symmetries
                const int block_i_inv = blockCountXY - (block_i + 1);
                const int columnBlocksToRight = (block_i_inv * (block_i_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_j + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                const int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;

                // cache part of rhs b in local memory
                nd_item.barrier();
                local_b[nd_item.get_local_id()] = b[block_j * matrixBlockSize + nd_item.get_local_id()];
                nd_item.barrier();

                // go through all columns of the block and compute the matrix vector product
                // the block in storage now has to be interpreted as transposed since we are working on the data of the symmetric block
                for (int j = 0; j < matrixBlockSize; ++j) {
                    resultValue += A[blockStartIndex + j * matrixBlockSize + iInBlock] * local_b[j];
                }
            }

            // store the result
            result[i] = resultValue;
        });
    });

    return event;
}

sycl::event MatrixVectorOperations::matrixVectorBlock_CPU(sycl::queue& queue, const conf::fp_type* A,
                                                          const conf::fp_type* b,
                                                          conf::fp_type* result, const int blockStart_i,
                                                          const int blockStart_j, const int blockCount_i,
                                                          const int blockCount_j, const int blockCountXY,
                                                          const bool reset) {
    // global range corresponds to number of rows in the (sub) matrix
    const std::size_t globalRange = blockCount_i * conf::matrixBlockSize;
    const auto kernelRange = range{globalRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    const bool addToPreviousEntries = !reset;


    sycl::event event = queue.submit([&](handler& h) {
        h.parallel_for(kernelRange, [=](auto& id) {
            // row i in the matrix
            const int i = id[0] + blockStart_i * matrixBlockSize;

            // local row index in current matrix block
            const int iInBlock = i % matrixBlockSize;

            // row index of matrix block current work-item will work with
            const int block_i = sycl::floor(static_cast<double>(i) / matrixBlockSize);

            // block count of all columns except the first one
            const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

            // block ID in the symmetric matrix
            int blockID = 0;

            // first index for block columns
            int block_j = blockStart_j;

            conf::fp_type resultValue = 0;
            if (addToPreviousEntries) {
                resultValue += result[i];
            }

            // First step: Process all matrix blocks up to the diagonal block (included) or the most left block that should be processed
            // the blocks can be interpreted as they are stored in memory
            for (; block_j <= min(block_i, blockStart_j + blockCount_j - 1); ++block_j) {
                // number of blocks in row to the right (if matrix would be full)
                const int block_j_inv = blockCountXY - (block_j + 1);

                // total number of blocks to the right that are stored
                const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_i + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;
                int rowStartIndex = blockStartIndex + iInBlock * matrixBlockSize;

                // go through all columns of the block and compute the matrix vector product
#pragma clang loop vectorize(enable)
                for (int j = 0; j < matrixBlockSize; ++j) {
                    resultValue += A[rowStartIndex + j] * b[block_j * matrixBlockSize + j];
                }
            }


            // Second step: Process all matrix blocks after the diagonal block
            // the blocks have to be interpreted as transposed, since the upper triangle is not stored explicitly
            for (; block_j < blockStart_j + blockCount_j; ++block_j) {
                // same block ID calculation as previously, but now block_i and block_j have to be swapped due to symmetries
                const int block_i_inv = blockCountXY - (block_i + 1);
                const int columnBlocksToRight = (block_i_inv * (block_i_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_j + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                const int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;

                // go through all columns of the block and compute the matrix vector product
                // the block in storage now has to be interpreted as transposed since we are working on the data of the symmetric block
#pragma clang loop vectorize(disable)
                for (int j = 0; j < matrixBlockSize; ++j) {
                    resultValue += A[blockStartIndex + j * matrixBlockSize + iInBlock] * b[block_j * matrixBlockSize +
                        j];
                }
            }

            // store the result
            result[i] = resultValue;
        });
    });

    return event;
}

sycl::event MatrixVectorOperations::triangularSolveVector(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b, int blockStart, int blockCount, int blockRow, int blockID, bool transposed) {
    // one work-group per rhs
    const range globalRange(conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    const int blockStartIndex = blockID * static_cast<int>(conf::matrixBlockSize) * static_cast<int>(conf::matrixBlockSize);

    const int N = conf::N;

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            int local_i = nd_item.get_local_id(0);

            // block in the matrix where the results are written
            const int blockStartIndex_B = blockRow * matrixBlockSize;

            const int blockStartIndex_L = blockStartIndex;

            for (int i = 0; i < matrixBlockSize; ++i) {
                int k = i;
                if (transposed) {
                    k = matrixBlockSize - (k + 1);
                }

                // b_k = b_k/a_kk
                const conf::fp_type b_k = b[blockStartIndex_B + k] / A[blockStartIndex_L + k * matrixBlockSize + k];

                nd_item.barrier();

                if (local_i == 0 && blockStartIndex_B + k < N) {
                    b[blockStartIndex_B + k] = b_k;
                }

                bool condition = (!transposed) ? local_i > k : local_i < k;

                if (condition && blockStartIndex_B + k < N) {
                    // b_i = b_i - a_ik*b_k
                    b[blockStartIndex_B + local_i] = b[blockStartIndex_B + local_i] - A[blockStartIndex_L + local_i * matrixBlockSize + k] * b_k;
                }

                nd_item.barrier();
            }
        });
    });

    return event;
}

