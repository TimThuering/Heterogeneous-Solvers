#include <sycl/sycl.hpp>

#include "MatrixOperations.hpp"
#include "SymmetricMatrix.hpp"

using namespace sycl;

void MatrixOperations::matrixVectorBlock(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b, conf::fp_type* result,
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
            const int i = nd_item.get_global_id() + blockStart_i;

            // local row index in current matrix block
            const int iInBlock = i % matrixBlockSize;

            // row index of matrix block current work-item will work with
            const int block_i = sycl::floor(static_cast<double>(i) / matrixBlockSize);

            // id of block in the matrix data structure for symmetric matrices
            int blockID = block_i;

            // keeps track how many (stored) blocks exists in columns to the left of the current column
            int blockCountLeftColumns = 0;

            // First step: Process all matrix blocks up to the diagonal block (included)
            // the blocks can be interpreted as they are stored in memory
            for (int block_j = blockStart_j; block_j <= block_i; ++block_j)
            {
                // startIndex of the current block with blockID in the symmetric matrix data structure
                int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;
                int rowStartIndex = blockStartIndex + iInBlock * matrixBlockSize;

                // go through all columns of the block and compute the matrix vector product
                for (int j = 0; j < matrixBlockSize; ++j)
                {
                    result[i] += A[rowStartIndex + j] * b[block_j * matrixBlockSize + j];
                }

                // calculate the new block ID
                blockID += blockCountXY - block_j - 1;
                blockCountLeftColumns += blockCountXY - block_j;
            }

            // new blockID is the ID of the diagonal block + 1
            blockID = blockCountLeftColumns - (blockCountXY - block_i) + 1;

            // Second step: Process all matrix blocks after the diagonal block
            // the blocks have to be interpreted as transposed, since the upper triangle is not stored explicitly
            for (int block_j = block_i + 1; block_j < blockStart_j + blockCount_j; ++block_j)
            {
                // go through all columns of the block and compute the matrix vector product
                // the block in storage now has to be interpreted as transposed since we are working on the data of the symmetric block
                for (int j = 0; j < matrixBlockSize; ++j)
                {
                    int blockStartIndex = blockID * matrixBlockSize * matrixBlockSize;
                    result[i] += A[blockStartIndex + j * matrixBlockSize + iInBlock] * b[block_j * matrixBlockSize + j];
                }
                // update the blockID
                blockID += 1;
            }
        });
    });
}
