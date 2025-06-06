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
                const conf::fp_type b_k = A[blockStartIndex_B + group_id_j * matrixBlockSize + k] / A[blockStartIndex_L
                    + k * matrixBlockSize + k];

                nd_item.barrier();

                if (local_i == 0) {
                    A[blockStartIndex_B + group_id_j * matrixBlockSize + k] = b_k;
                }

                if (local_i > k) {
                    // b_i = b_i - a_ik*b_k
                    A[blockStartIndex_B + group_id_j * matrixBlockSize + local_i] =
                        A[blockStartIndex_B + group_id_j * matrixBlockSize + local_i] -
                        A[blockStartIndex_L + local_i * matrixBlockSize + k] * b_k;
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
            conf::fp_type value = A[blockStartIndex_B + group_id_j * matrixBlockSize + local_i];

            // b_0 has to available for all work-items in the next iteration
            if (local_i == 0) {
                local_column[0] = value * diagonal_ii;
            }
            nd_item.barrier();

            // loop over columns in the lower triangular matrix
            for (int k = 0; k < matrixBlockSize; ++k) {
                if (local_i > k) {
                    // b_i = b_i - a_ik*b_k
                    value = value - A[blockStartIndex_L + local_i * matrixBlockSize + k] * local_column[k];
                    if (local_i == k + 1) {
                        // make b_{k+1} available to all work-items for the next iteration
                        local_column[local_i] = value * diagonal_ii;
                    }
                }

                // synchronize so that all work-items see b_{k+1} in the next iteration
                nd_item.barrier();
            }

            // store final value in global memory, also works for last entry since value * diagonal_ii is recomputed
            A[blockStartIndex_B + group_id_j * matrixBlockSize + local_i] = value * diagonal_ii;
        });
    });

    return event;
}

sycl::event MatrixMatrixOperations::triangularSolve_optimizedCPU(sycl::queue& queue, conf::fp_type* A,
                                                                 const int blockID,
                                                                 const int blockRow, const int blockStart,
                                                                 const int blockCount) {
    // one work-group per rhs
    const range globalRange(blockCount, conf::matrixBlockSize);

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    const int blockStartIndex = blockID * static_cast<int>(conf::matrixBlockSize) * static_cast<int>(
        conf::matrixBlockSize);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(globalRange, [=](auto& id) {
            const int group_id_i = id[0];
            const int group_id_j = id[1];

            // block in the matrix where the results are written
            const int block_id_B = blockID + (blockStart - blockRow) + group_id_i;
            const int blockStartIndex_B = block_id_B * matrixBlockSize * matrixBlockSize;

            const int blockStartIndex_L = blockStartIndex;

#pragma unroll
            for (int k = 0; k < matrixBlockSize; ++k) {
                // b_k = b_k/a_kk
                const conf::fp_type b_k = A[blockStartIndex_B + group_id_j * matrixBlockSize + k] /
                    A[blockStartIndex_L + k * matrixBlockSize + k];

                A[blockStartIndex_B + group_id_j * matrixBlockSize + k] = b_k;

#pragma clang loop vectorize(enable) unroll(enable)
                for (int j = k + 1; j < matrixBlockSize; ++j) {
                    // b_i = b_i - a_ik*b_k
                    A[blockStartIndex_B + group_id_j * matrixBlockSize + j] =
                        A[blockStartIndex_B + group_id_j * matrixBlockSize + j] -
                        A[blockStartIndex_L + j * matrixBlockSize + k] * b_k;
                }
            }
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

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int local_j = nd_item.get_local_id(1);
            const int group_id_i = nd_item.get_group().get_group_id(0);
            const int group_id_j = nd_item.get_group().get_group_id(1);

            // block offset of current work group in column direction
            const int columnOffset = (blockStart - blockRow) + (group_id_i / wgCount_xy);

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

            if (i >= j) {
                // perform update for lower triangle of the diagonal
                for (int k = 0; k < matrixBlockSize; ++k) {
                    // B_diag = B_diag - B_col * B_col^T
                    A[blockStartIndex_diag + i * matrixBlockSize + j] =
                        A[blockStartIndex_diag + i * matrixBlockSize + j] -
                        A[blockStartIndex_col + i * matrixBlockSize + k] *
                        A[blockStartIndex_col + j * matrixBlockSize + k];
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

    sycl::event event = queue.submit([&](sycl::handler& h) {
        auto local_tile_A = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);
        auto local_tile_B = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int local_j = nd_item.get_local_id(1);
            const int group_id_i = nd_item.get_group().get_group_id(0);
            const int group_id_j = nd_item.get_group().get_group_id(1);

            // block offset of current work group in column direction
            const int columnOffset = (blockStart - blockRow) + (group_id_i / wgCount_xy);

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

            const int group_mod_count_i = (group_id_i % wgCount_xy);
            const int group_mod_count_j = (group_id_j % wgCount_xy);

            // indices in of the current work-item in the matrix block
            const int internalBlockOffset_i = group_mod_count_i * wgSize_xy;
            const int internalBlockOffset_j = group_mod_count_j * wgSize_xy;

            const int i = internalBlockOffset_i + local_i;
            const int j = internalBlockOffset_j + local_j;

            // load initial value for result
            conf::fp_type value = 0.0;
            if (i >= j) {
                value = A[blockStartIndex_diag + i * matrixBlockSize + j];
            }

            // perform update for lower triangle of the diagonal
            for (int t = 0; t < matrixBlockSize / wgSize_xy; ++t) {
                // if tile is below diagonal or a diagonal tile, cache it in local memory
                if (i >= j || group_mod_count_i == group_mod_count_j) {
                    // normal block
                    local_tile_A[local_i][local_j] = A[blockStartIndex_col + i * matrixBlockSize + t * wgSize_xy +
                        local_j];

                    // transposed block
                    local_tile_B[local_i][local_j] = A[blockStartIndex_col + j * matrixBlockSize + t * wgSize_xy +
                        local_i];
                }
                group_barrier(nd_item.get_group(), memory_scope::work_group);

                if (i >= j) {
                    for (int k = 0; k < wgSize_xy; ++k) {
                        // B_diag = B_diag - B_col * B_col^T
                        value = value - local_tile_A[local_i][k] * local_tile_B[k][local_j];
                    }
                }
                group_barrier(nd_item.get_group(), memory_scope::work_group);
            }
            if (i >= j) {
                A[blockStartIndex_diag + i * matrixBlockSize + j] = value;
            }
        });
    });

    return event;
}


sycl::event MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedCPU(sycl::queue& queue, conf::fp_type* A,
                                                                               const int blockID,
                                                                               const int blockRow, const int blockStart,
                                                                               const int blockCount,
                                                                               const int blockCountXY) {
    const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    // block count of all columns except the first one
    const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(globalRange, [=](auto& id) {
            const int local_i = id[1];
            const int local_j = id[0];

            const int group_id = local_j / matrixBlockSize;

            // block offset of current work group in column direction
            const int columnOffset = (blockStart - blockRow) + group_id;

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

            const int i = local_i;
            const int j = local_j % matrixBlockSize;

            if (i >= j) {
                conf::fp_type value = A[blockStartIndex_diag + i * matrixBlockSize + j];
                // perform update for lower triangle of the diagonal
#pragma clang loop vectorize(enable) unroll(enable)
                for (int k = 0; k < matrixBlockSize; ++k) {
                    // B_diag = B_diag - B_col * B_col^T
                    value = value - A[blockStartIndex_col + i * matrixBlockSize + k] *
                        A[blockStartIndex_col + j * matrixBlockSize + k];
                }
                A[blockStartIndex_diag + i * matrixBlockSize + j] = value;
            }
        });
    });

    return event;
}


sycl::event MatrixMatrixOperations::matrixMatrixStep(sycl::queue& queue, conf::fp_type* A, const int blockID,
                                                     const int blockRow,
                                                     const int blockStart, const int blockCount,
                                                     const int blockCountXY) {
    const int wgSize_xy = conf::workGroupSizeGEMM_xy;
    if (conf::matrixBlockSize % wgSize_xy != 0) {
        throw std::runtime_error("xy work-group dimension for matrix multiplication must divide matrix block size");
    }

    const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

    const int rowsAbove = blockStart - (blockRow + 2);
    const int rowsBelow = blockCountXY - blockStart - blockCount;

    // block Count including rows above and below that should not be processed
    const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

    const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
    const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
    const int lowerBlockCount = totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

    const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

    const range globalRange(wgCount_xy * wgSize_xy, wgCount * wgCount_xy * wgSize_xy);
    const range localRange(wgSize_xy, wgSize_xy);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int local_j = nd_item.get_local_id(1);
            const int group_id_i = nd_item.get_group().get_group_id(1);
            const int group_id_j = nd_item.get_group().get_group_id(0);

            // block ID of matrix blocks if one would enumerate them row by row
            const int rowBlockID = upperBlockCount + (group_id_i / wgCount_xy);

            // row ID in the lower triangle where the computation takes place
            const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

            const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

            // column ID of the matrix block int the lower triangle the current work-group is associated with
            const int columnID = rowBlockID - blocksAboveCurrentRow;

            // calculation of the block ID of matrix block associated with this work-group
            const int wgBlockID_A = blockID + blockCountXY - blockRow + columnID + 1 + (totalBlockCount - ((blockCountXY
                - blockRow - 2 - columnID) * (blockCountXY - blockRow - 2 - columnID + 1) / 2)) + rowID - columnID;

            const int wgBlockID_B = blockID + rowID + 2;
            const int wgBlockID_C = blockID + columnID + 1;

            const int blockStartIndex_A = wgBlockID_A * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_B = wgBlockID_B * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_C = wgBlockID_C * matrixBlockSize * matrixBlockSize;

            // indices in of the current work-item in the matrix block
            const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
            const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

            const int i = internalBlockOffset_i + local_i;
            const int j = internalBlockOffset_j + local_j;

            for (int k = 0; k < matrixBlockSize; ++k) {
                // A = A - B * C^T
                A[blockStartIndex_A + i * matrixBlockSize + j] =
                    A[blockStartIndex_A + i * matrixBlockSize + j] -
                    A[blockStartIndex_B + i * matrixBlockSize + k] *
                    A[blockStartIndex_C + j * matrixBlockSize + k];
            }
        });
    });

    return event;
}

sycl::event MatrixMatrixOperations::matrixMatrixStep_optimizedGPU(sycl::queue& queue, conf::fp_type* A, int blockID,
                                                                  int blockRow, int blockStart, int blockCount,
                                                                  int blockCountXY) {
    const int wgSize_xy = conf::workGroupSizeGEMM_xy;
    if (conf::matrixBlockSize % wgSize_xy != 0) {
        throw std::runtime_error("xy work-group dimension for matrix multiplication must divide matrix block size");
    }

    const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

    const int rowsAbove = blockStart - (blockRow + 2);
    const int rowsBelow = blockCountXY - blockStart - blockCount;

    // block Count including rows above and below that should not be processed
    const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

    const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
    const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
    const int lowerBlockCount = totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

    const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

    const range globalRange(wgCount_xy * wgSize_xy, wgCount * wgCount_xy * wgSize_xy);
    const range localRange(wgSize_xy, wgSize_xy);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        auto local_tile_B = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);
        auto local_tile_C = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy + 1), h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);
            const int local_j = nd_item.get_local_id(1);
            const int group_id_i = nd_item.get_group().get_group_id(1);
            const int group_id_j = nd_item.get_group().get_group_id(0);

            // block ID of matrix blocks if one would enumerate them row by row
            const int rowBlockID = upperBlockCount + (group_id_i / wgCount_xy);

            // row ID in the lower triangle where the computation takes place
            const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

            const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

            // column ID of the matrix block int the lower triangle the current work-group is associated with
            const int columnID = rowBlockID - blocksAboveCurrentRow;

            // calculation of the block ID of matrix block associated with this work-group
            const int wgBlockID_A = blockID + blockCountXY - blockRow + columnID + 1 + (totalBlockCount - ((blockCountXY
                - blockRow - 2 - columnID) * (blockCountXY - blockRow - 2 - columnID + 1) / 2)) + rowID - columnID;

            const int wgBlockID_B = blockID + rowID + 2;
            const int wgBlockID_C = blockID + columnID + 1;

            const int blockStartIndex_A = wgBlockID_A * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_B = wgBlockID_B * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_C = wgBlockID_C * matrixBlockSize * matrixBlockSize;

            // indices in of the current work-item in the matrix block
            const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
            const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

            const int i = internalBlockOffset_i + local_i;
            const int j = internalBlockOffset_j + local_j;


            // i coordinate for matrix c that needs to be interpreted as transposed later but is loaded non-transposed
            const int i_c = internalBlockOffset_j + local_i;

            // load initial value for result
            conf::fp_type value = A[blockStartIndex_A + i * matrixBlockSize + j];

            const int startIndexB = blockStartIndex_B + i * matrixBlockSize + local_j;
            const int startIndexC = blockStartIndex_C + i_c * matrixBlockSize + local_j;

            // perform update for lower triangle of the diagonal
            for (int t = 0; t < wgCount_xy; ++t) {
                // normal block
                local_tile_B[local_i][local_j] = A[startIndexB + t * wgSize_xy];

                // transposed block
                local_tile_C[local_i][local_j] = A[startIndexC + t * wgSize_xy];

                group_barrier(nd_item.get_group(), memory_scope::work_group);


#pragma unroll
                for (int k = 0; k < wgSize_xy; ++k) {
                    // B_diag = B_diag - B_col * B_col^T
                    value = value - local_tile_B[local_i][k] * local_tile_C[local_j][k];
                }
                group_barrier(nd_item.get_group(), memory_scope::work_group);
            }
            A[blockStartIndex_A + i * matrixBlockSize + j] = value;
        });
    });

    return event;
}


sycl::event MatrixMatrixOperations::matrixMatrixStep_optimizedCPU(sycl::queue& queue, conf::fp_type* A,
                                                                  const int blockID,
                                                                  const int blockRow,
                                                                  const int blockStart, const int blockCount,
                                                                  const int blockCountXY) {
    const int rowsAbove = blockStart - (blockRow + 2);
    const int rowsBelow = blockCountXY - blockStart - blockCount;

    // block Count including rows above and below that should not be processed
    const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

    const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
    const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
    const int lowerBlockCount = totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

    const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

    const range globalRange(wgCount * conf::matrixBlockSize, conf::matrixBlockSize);

    const int matrixBlockSize = static_cast<int>(conf::matrixBlockSize);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(globalRange, [=](auto& id) {
            const int local_i = id[1];
            const int local_j = id[0];

            const int group_id = local_j / matrixBlockSize;

            // block ID of matrix blocks if one would enumerate them row by row
            const int rowBlockID = upperBlockCount + group_id;

            // row ID in the lower triangle where the computation takes place
            const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

            const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

            // column ID of the matrix block int the lower triangle the current work-group is associated with
            const int columnID = rowBlockID - blocksAboveCurrentRow;

            // calculation of the block ID of matrix block associated with this work-group
            const int wgBlockID_A = blockID + blockCountXY - blockRow + columnID + 1 + (totalBlockCount - ((blockCountXY
                - blockRow - 2 - columnID) * (blockCountXY - blockRow - 2 - columnID + 1) / 2)) + rowID - columnID;

            const int wgBlockID_B = blockID + rowID + 2;
            const int wgBlockID_C = blockID + columnID + 1;

            const int blockStartIndex_A = wgBlockID_A * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_B = wgBlockID_B * matrixBlockSize * matrixBlockSize;
            const int blockStartIndex_C = wgBlockID_C * matrixBlockSize * matrixBlockSize;


            const int i = local_i;
            const int j = local_j % matrixBlockSize;

            conf::fp_type value = A[blockStartIndex_A + i * matrixBlockSize + j];
#pragma clang loop vectorize(enable) unroll(enable)
            for (int k = 0; k < matrixBlockSize; ++k) {
                // A = A - B * C^T
                value = value - A[blockStartIndex_B + i * matrixBlockSize + k] * A[blockStartIndex_C + j *
                    matrixBlockSize + k];
            }

            A[blockStartIndex_A + i * matrixBlockSize + j] = value;
        });
    });

    return event;
}
