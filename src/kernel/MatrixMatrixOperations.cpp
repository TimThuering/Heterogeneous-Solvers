#include "MatrixMatrixOperations.hpp"

using namespace sycl;

sycl::event MatrixMatrixOperations::triangularSolve(sycl::queue& queue, conf::fp_type* A, const int blockID,
                                                    const int blockRow, const int blockCount) {
    // number of blocks the triangular solve has to be applied to
    // const int blockCount = static_cast<int>(conf::N) - 1 - blockRow;

    // one work-group per rhs
    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize, 1);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    const int blockStartIndex = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

    const long N = static_cast<long>(conf::N);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int group_id_i = nd_item.get_group().get_group_id(0);
            const int group_id_j = nd_item.get_group().get_group_id(1);

            // block in the matrix where the results are written
            const int block_id_B = blockRow + group_id_i + 1;
            const int blockStartIndex_B = block_id_B * matrixBlockSize * matrixBlockSize;

            // block with the triangular matrix
            const int block_id_L = blockID;
            const int blockStartIndex_L = blockStartIndex;

            // if (local_i == 0) {
            //     printf("%i,%i\n", group_id_i, group_id_j);
            // }

            // conf::fp_type b_i = A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j];

            for (int k = 0; k < matrixBlockSize; ++k) {
                // b_k = b_k/a_kk
                const conf::fp_type b_k = A[blockStartIndex_B + k * matrixBlockSize + group_id_j] / A[blockStartIndex_L + k * matrixBlockSize + k];

                nd_item.barrier();

                if (local_i == 0) {
                    A[blockStartIndex_B + k * matrixBlockSize + group_id_j] = b_k;
                }

                if (local_i > k) {
                    // b_i = b_i - a_ik*b_k
                    if ((blockRow * matrixBlockSize + local_i) < N) {
                        A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j] = A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j] - A[blockStartIndex_L + local_i * matrixBlockSize + k] * b_k;
                    }
                }

                nd_item.barrier();

            }

            // A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j] = b_i;
            // A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j] = local_i;
        });
    });

    return event;
}
