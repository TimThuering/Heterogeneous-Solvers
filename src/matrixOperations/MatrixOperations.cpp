#include <sycl/sycl.hpp>

#include "MatrixOperations.hpp"
#include "SymmetricMatrix.hpp"

using namespace sycl;

void MatrixOperations::matrixVectorBlock(queue& queue, conf::fp_type* A, conf::fp_type* b, conf::fp_type* result,
                                         const int blockStart_i,
                                         const int blockStart_j, const int blockCount_i, const int blockCount_j,
                                         const int blockCountXY)
{
    // global range corresponds to number of rows in the (sub) matrix
    const range globalRange(blockCount_i * conf::matrixBlockSize);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;


    queue.submit([&](handler& h)
    {
        h.parallel_for(kernelRange, [=](auto& nd_item)
        {
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

            // First step: Process all matrix blocks up to the diagonal block (included) or the most left block that should be processed
            // the blocks can be interpreted as they are stored in memory
            for (; block_j <= min(block_i, blockStart_j + blockCount_j - 1); ++block_j)
            {
                // number of blocks in row to the right (if matrix would be full)
                int block_j_inv = blockCountXY - (block_j + 1);

                // total number of blocks to the right that are stored
                int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_i + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;
                int rowStartIndex = blockStartIndex + iInBlock * matrixBlockSize;

                // go through all columns of the block and compute the matrix vector product
                for (int j = 0; j < matrixBlockSize; ++j)
                {
                    result[i] += A[rowStartIndex + j] * b[block_j * matrixBlockSize + j];
                }
            }


            // Second step: Process all matrix blocks after the diagonal block
            // the blocks have to be interpreted as transposed, since the upper triangle is not stored explicitly
            for (; block_j < blockStart_j + blockCount_j; ++block_j)
            {
                // same block ID calculation as previously, but now block_i and block_j have to be swapped due to symmetries
                int block_i_inv = blockCountXY - (block_i + 1);
                int columnBlocksToRight = (block_i_inv * (block_i_inv + 1)) / 2;

                // id of block in the matrix data structure for symmetric matrices
                blockID = block_j + referenceBlockCount - columnBlocksToRight;

                // startIndex of the current block with blockID in the symmetric matrix data structure
                int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;

                // go through all columns of the block and compute the matrix vector product
                // the block in storage now has to be interpreted as transposed since we are working on the data of the symmetric block
                for (int j = 0; j < matrixBlockSize; ++j)
                {
                    result[i] += A[blockStartIndex + j * matrixBlockSize + iInBlock] * b[block_j * matrixBlockSize + j];
                }

            }
        });
    });
}
