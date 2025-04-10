#include <sycl/sycl.hpp>

#include "MatrixOperations.hpp"
#include "SymmetricMatrix.hpp"

using namespace sycl;

void MatrixOperations::matrixVectorBlock(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b,
                                         const int blockStart_i,
                                         const int blockStart_j, const int blockCount_i, const int blockCount_j)
{
    const range globalRange(blockCount_i);
    const range localRange(conf::workGroupSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    queue.submit([&](handler& h)
    {
        h.parallel_for(kernelRange, [=](auto& nd_item)
        {
            const int i = nd_item.get_global_id();

            // const int matrix_i = blockStart_i * matrixBlockSize * matrixBlockSize + i;

            A[0] = i;

        });
    });
}
