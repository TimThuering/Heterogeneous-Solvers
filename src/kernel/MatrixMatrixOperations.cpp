#include "MatrixMatrixOperations.hpp"

using namespace sycl;

sycl::event MatrixMatrixOperations::triangularSolve(sycl::queue& queue, conf::fp_type* A, const int blockID,
                                                    const int blockRow, const int blockStart, const int blockCount) {
    // one work-group per rhs
    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize, 1);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    const int blockStartIndex = blockID * static_cast<int>(conf::matrixBlockSize) * static_cast<int>(
        conf::matrixBlockSize);

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

sycl::event MatrixMatrixOperations::triangularSolve_optimizedGPU(sycl::queue& queue, conf::fp_type* A,
                                                                 const int blockID,
                                                                 const int blockRow, const int blockStart,
                                                                 const int blockCount) {
    // one work-group per rhs
    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize, 1);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    const int blockStartIndex = blockID * static_cast<int>(conf::matrixBlockSize) * static_cast<int>(
        conf::matrixBlockSize);

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

sycl::event MatrixMatrixOperations::symmetricMatrixMatrixDiagonal(sycl::queue& queue, conf::fp_type* A,
                                                                  const int blockID,
                                                                  const int blockRow, const int blockStart,
                                                                  const int blockCount, const int blockCountXY) {
    const int wgSize_xy = conf::workGroupSizeGEMM_xy;
    if (conf::matrixBlockSize % wgSize_xy != 0) {
        throw std::runtime_error("xy work-group dimension for matrix multiplication must divide matrix block size");
    }

    const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
    const range localRange(wgSize_xy, wgSize_xy);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    // block count of all columns except the first one
    const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

    const long N = static_cast<long>(conf::N);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int local_j = nd_item.get_local_id(1);
            const int group_id_i = nd_item.get_group().get_group_id(0);
            const int group_id_j = nd_item.get_group().get_group_id(1);

            // block offset of current work group in column direction
            const int columnOffset = blockStart + (group_id_i / wgCount_xy);

            // x/y block coordinate of the diagonal block processed by this work-group
            const int blockXYIndexDiagonal = blockRow + columnOffset;

            // number of blocks in row to the right (if matrix would be full)
            const int block_j_inv = blockCountXY - (blockXYIndexDiagonal + 1);

            // total number of blocks to the right that are stored
            const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

            // id of block in the matrix data structure for symmetric matrices
            const int blockID_wg_diag = blockXYIndexDiagonal + referenceBlockCount - columnBlocksToRight;

            // id of the column block used in the matrix-matrix multiplication
            const int blockID_wg_col = blockID + columnOffset;

            // start indices of blocks involved in the syrk update
            const int blockStartIndex_diag = blockID_wg_diag * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_col = blockID_wg_col * matrixBlockSize * matrixBlockSize;

            // indices in of the current work-item in the matrix block
            const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
            const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

            const int i = internalBlockOffset_i + local_i;
            const int j = internalBlockOffset_j + local_j;

            // printf("%i,%i: %i,%i; %i,%i,  ---  %i,%i\n", group_id_i, group_id_j, local_i, local_j, i, j, blockID_wg_col, blockID_wg_diag);
            // printf("%i,%i: %i,%i; %i,%i,  ---  %i,%i\n", group_id_i, group_id_j, local_i, local_j, i, j, internalBlockOffset_i, internalBlockOffset_j);

            if (i >= j) {
                // perform update for lower triangle of the diagonal
                for (int k = 0; k < matrixBlockSize; ++k) {
                    // B_diag = B_diag - B_col * B_col^T
                    A[blockStartIndex_diag + i * matrixBlockSize + j] = A[blockStartIndex_diag + i * matrixBlockSize +
                        j] - A[blockStartIndex_col + i * matrixBlockSize +
                        k] * A[blockStartIndex_col + j * matrixBlockSize + k];
                }
            }
        });
    });

    return event;
}

sycl::event MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedGPU(sycl::queue& queue, conf::fp_type* A,
                                                                               int blockID, int blockRow,
                                                                               int blockStart, int blockCount,
                                                                               int blockCountXY) {
    const int wgSize_xy = conf::workGroupSizeGEMM_xy;
    if (conf::matrixBlockSize % wgSize_xy != 0) {
        throw std::runtime_error("xy work-group dimension for matrix multiplication must divide matrix block size");
    }

    const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
    const range localRange(wgSize_xy, wgSize_xy);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    // block count of all columns except the first one
    const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

    const long N = static_cast<long>(conf::N);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        auto local_tile_A = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);
        auto local_tile_B = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int local_j = nd_item.get_local_id(1);
            const int group_id_i = nd_item.get_group().get_group_id(0);
            const int group_id_j = nd_item.get_group().get_group_id(1);

            // block offset of current work group in column direction
            const int columnOffset = blockStart + (group_id_i / wgCount_xy);

            // x/y block coordinate of the diagonal block processed by this work-group
            const int blockXYIndexDiagonal = blockRow + columnOffset;

            // number of blocks in row to the right (if matrix would be full)
            const int block_j_inv = blockCountXY - (blockXYIndexDiagonal + 1);

            // total number of blocks to the right that are stored
            const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

            // id of block in the matrix data structure for symmetric matrices
            const int blockID_wg_diag = blockXYIndexDiagonal + referenceBlockCount - columnBlocksToRight;

            // id of the column block used in the matrix-matrix multiplication
            const int blockID_wg_col = blockID + columnOffset;

            // start indices of blocks involved in the syrk update
            const int blockStartIndex_diag = blockID_wg_diag * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_col = blockID_wg_col * matrixBlockSize * matrixBlockSize;

            // indices in of the current work-item in the matrix block
            const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
            const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

            const int i = internalBlockOffset_i + local_i;
            const int j = internalBlockOffset_j + local_j;

            conf::fp_type value = A[blockStartIndex_diag + i * matrixBlockSize + j];

            // perform update for lower triangle of the diagonal
            for (int t = 0; t < matrixBlockSize / wgSize_xy; ++t) {
                local_tile_A[local_i][local_j] = A[blockStartIndex_col + i * matrixBlockSize + t * wgSize_xy + local_j];
                local_tile_B[local_i][local_j] = A[blockStartIndex_col + j * matrixBlockSize + t *  wgSize_xy + local_i];
                // if (group_id_i == 0 && group_id_j == 0) {
                //     printf("%i,%i: %f %f\n",local_i, local_j, local_tile_B[local_j][0], A[blockStartIndex_col + j * matrixBlockSize + 0]);
                // }
                nd_item.barrier();
                if (i >= j) {
                    for (int k = 0; k < wgSize_xy; ++k) {
                        // B_diag = B_diag - B_col * B_col^T
                        value = value - local_tile_A[local_i][k] * local_tile_B[k][local_j];
                    }
                }
                nd_item.barrier();
            }
            A[blockStartIndex_diag + i * matrixBlockSize + j] = value;
        });
    });

    return event;
}
