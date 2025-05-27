#include "MatrixMatrixOperations.hpp"

using namespace sycl;

sycl::event MatrixMatrixOperations::triangularSolve(sycl::queue& queue, conf::fp_type* A, const int blockID,
                                                    const int blockRow, const int blockStart, const int blockCount) {
    // one work-group per rhs
    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize, 1);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    const int blockStartIndex = blockID * static_cast<int>(conf::matrixBlockSize) * static_cast<int>(conf::matrixBlockSize);

    const long N = static_cast<long>(conf::N);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int group_id_i = nd_item.get_group().get_group_id(0);
            const int group_id_j = nd_item.get_group().get_group_id(1);

            // block in the matrix where the results are written
            const int block_id_B = blockID + (blockStart - blockRow) + group_id_i;
            const int blockStartIndex_B = block_id_B * matrixBlockSize * matrixBlockSize;

            const int blockStartIndex_L = blockStartIndex;

            for (int k = 0; k < matrixBlockSize; ++k) {
                // b_k = b_k/a_kk
                const conf::fp_type b_k = A[blockStartIndex_B + k * matrixBlockSize + group_id_j] / A[blockStartIndex_L
                    + k * matrixBlockSize + k];

                nd_item.barrier();

                if (local_i == 0) {
                    A[blockStartIndex_B + k * matrixBlockSize + group_id_j] = b_k;
                }

                if (local_i > k) {
                    // b_i = b_i - a_ik*b_k
                    if (((blockStart + group_id_i) * matrixBlockSize + local_i) < N) {
                        A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j] = A[blockStartIndex_B + local_i *
                            matrixBlockSize + group_id_j] - A[blockStartIndex_L + local_i * matrixBlockSize + k] * b_k;
                    }
                }

                nd_item.barrier();
            }
        });
    });

    return event;
}

sycl::event MatrixMatrixOperations::triangularSolve_optimizedGPU(sycl::queue& queue, conf::fp_type* A, const int blockID,
                                                              const int blockRow, const int blockStart,
                                                              const int blockCount) {
    // one work-group per rhs
    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize, 1);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    const int blockStartIndex = blockID * static_cast<int>(conf::matrixBlockSize) * static_cast<int>(conf::matrixBlockSize);

    const long N = static_cast<long>(conf::N);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        auto local_column = local_accessor<conf::fp_type, 1>(matrixBlockSize, h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int group_id_i = nd_item.get_group().get_group_id(0);
            const int group_id_j = nd_item.get_group().get_group_id(1);

            // block in the matrix where the results are written
            const int block_id_B = blockID + (blockStart - blockRow) + group_id_i;
            const int blockStartIndex_B = block_id_B * matrixBlockSize * matrixBlockSize;

            const int blockStartIndex_L = blockStartIndex;

            // inverse of diagonal value in lower triangular matrix
            const conf::fp_type diagonal_ii = 1.0 / A[blockStartIndex_L + local_i * matrixBlockSize + local_i];

            // current value of the position in the column that will be updated by the work-item
            conf::fp_type value = A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j];

            // b_0 has to available for all work-items in the next iteration
            if (local_i == 0) {
                local_column[0] = value * diagonal_ii;
            }
            nd_item.barrier();

            // loop over columns in the lower triangular matrix
            for (int k = 0; k < matrixBlockSize; ++k) {
                if (local_i > k) {
                    if (((blockStart + group_id_i) * matrixBlockSize + local_i) < N) {
                        // b_i = b_i - a_ik*b_k
                        value = value - A[blockStartIndex_L + local_i * matrixBlockSize + k] * local_column[k];
                        if (local_i == k + 1) {
                            // make b_{k+1} available to all work-items for the next iteration
                            local_column[local_i] = value * diagonal_ii;
                        }
                    }
                }

                // synchronize so that all work-items see b_{k+1} in the next iteration
                nd_item.barrier();
            }

            // store final value in global memory, also works for last entry since value * diagonal_ii is recomputed
            A[blockStartIndex_B + local_i * matrixBlockSize + group_id_j] = value * diagonal_ii;
        });
    });

    return event;
}

sycl::event MatrixMatrixOperations::symmetricMatrixMatrixDiagonal(sycl::queue& queue, conf::fp_type* A, int blockID,
    int blockRow, int blockStart, int blockCount) {
}
