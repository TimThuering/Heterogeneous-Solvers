#include "Cholesky.hpp"

#include "MatrixMatrixOperations.hpp"
#include "MatrixOperations.hpp"
#include "MatrixParser.hpp"
#include "UtilityFunctions.hpp"

Cholesky::Cholesky(SymmetricMatrix& A, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer):
    A(A),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue),
    loadBalancer(std::move(loadBalancer)) {
}

void Cholesky::initGPUMemory() {
    // init GPU data structure for matrix A on which the Cholesky decomposition should be performed
    A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);
    // Copy the matrix A to the GPU
    gpuQueue.submit([&](handler& h) {
        h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
    }).wait();
    executionTimes.endMemoryInitGPU = std::chrono::steady_clock::now();
}

void Cholesky::initExecutionTimes() {
    executionTimes.startColumn = std::chrono::steady_clock::now();

    executionTimes.startCholesky = std::chrono::steady_clock::now();
    executionTimes.endCholesky = executionTimes.startCholesky;

    executionTimes.startCopy_row = std::chrono::steady_clock::now();
    executionTimes.endCopy_row = executionTimes.startCopy_row;

    executionTimes.startCopy_column = std::chrono::steady_clock::now();
    executionTimes.endCopy_column = executionTimes.startCopy_column;

    executionTimes.startTriangularSolve = std::chrono::steady_clock::now();
    executionTimes.endTriangularSolve = executionTimes.startTriangularSolve;

    executionTimes.startMatrixMatrixDiagonal = std::chrono::steady_clock::now();
    executionTimes.endMatrixMatrixDiagonal = executionTimes.startMatrixMatrixDiagonal;

    executionTimes.startMatrixMatrix = std::chrono::steady_clock::now();
    executionTimes.endMatrixMatrix = executionTimes.startMatrixMatrix;
}

void Cholesky::shiftSplitRowComm(const int blockCountATotal, const std::size_t blockSizeBytes, const int k) {
    executionTimes.startCopy_row = std::chrono::steady_clock::now();

    std::cout << "Copying Blocks" << std::endl;

    // copy part of each column that was computed on the GPU and now has to be computed by the CPU after re-balancing
    for (int c = 0; c < blockCountCPU + k + 1; ++c) {
        const int columnsToRight = A.blockCountXY - c;
        // first block in the column updated by the GPU
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2) + std::max(A.blockCountXY - c - blockCountGPU, 0);

        const std::size_t blockStartIndexFirstGPUBlock = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

        gpuQueue.submit([&](handler& h) {
            h.memcpy(&A.matrixData[blockStartIndexFirstGPUBlock], &A_gpu[blockStartIndexFirstGPUBlock], blockSizeBytes);
        });
    }
    gpuQueue.wait();

    executionTimes.endCopy_row = std::chrono::steady_clock::now();
}

void Cholesky::shiftSplit(const int blockCountATotal, const std::size_t blockSizeBytes, const int k) {
    if (k % loadBalancer->updateInterval != 0 || k == 0) {
        // normal iteration: move split downwards as usual to keep proportion between GPU and CPU similar

        // compute new block counts to keep CPU/GPU proportion similar throughout the algorithm
        const int newBlockCountGPU = std::min(std::max(static_cast<int>(std::ceil((A.blockCountXY - 1 - k) * gpuProportion)), minBlockCountGPU), A.blockCountXY - (k + 1));
        const int newBlockCountCPU = std::max(A.blockCountXY - 1 - k - newBlockCountGPU, 0);
        const int newBlockStartGPU = std::max(A.blockCountXY - newBlockCountGPU, k + 1);

        // true if block counts changed and the row that splits the matrix into a CPU and a GPU part has to change
        if (blockCountGPU != newBlockCountGPU && newBlockCountGPU >= minBlockCountGPU && gpuProportion != 1 && gpuProportion != 0) {
            // split has shifted --> communicate the row that was previously held by the GPU and now belongs to the CPU
            shiftSplitRowComm(blockCountATotal, blockSizeBytes, k);
        }

        blockCountGPU = newBlockCountGPU;
        blockCountCPU = newBlockCountCPU;
        blockStartGPU = newBlockStartGPU;
    } else {
        // iteration with re-balancing: move split up or down by possibly multiple rows, depending on new requested load distribution

        // const double gpuProportion_new = loadBalancer->getNewProportionGPU(metricsTracker);
        const double gpuProportion_new = 0.8;

        // compute new block counts for new requested GPU proportion
        const int blockCountGPU_new = std::min(std::max(static_cast<int>(std::ceil((A.blockCountXY - 1 - k) * gpuProportion_new)), minBlockCountGPU), A.blockCountXY - (k + 1));
        const int blockCountCPU_new = std::max(A.blockCountXY - 1 - k - blockCountGPU_new, 0);
        const int blockStartGPU_new = std::max(A.blockCountXY - blockCountGPU_new, k + 1);

        // TODO include threshold again. normal shift has to be done anyways
        if (blockCountGPU_new > blockCountGPU + conf::blockUpdateThreshold || blockCountCPU_new > blockCountCPU + conf::blockUpdateThreshold) {
            if (blockCountGPU_new > blockCountGPU) {
                // GPU proportion increased --> move split up
                const int additionalBlocks =  blockStartGPU - blockStartGPU_new;

                std::cout << "---- Re-balancing: ----" << std::endl;
                std::cout << "Shifting CPU block count from " << blockCountCPU << " to " << blockCountCPU_new << std::endl;


                // copy part of each column that was computed on the CPU and now has to be computed by the GPU after re-balancing
                for (int c = 0; c < blockCountCPU_new + k + 2 + (additionalBlocks - 1); ++c) {
                    const int columnsToRight = A.blockCountXY - c;
                    // first block in the column updated by the GPU
                    int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2) ;

                    int blocksToCopy = additionalBlocks;

                    if (c < blockCountCPU_new + k + 1) {
                        blockID += std::max(A.blockCountXY - c - (blockCountGPU + additionalBlocks), 0);
                    } else {
                        blocksToCopy = additionalBlocks - (c - (blockCountCPU_new + k + 1));
                    }

                    std::cout << blockID << " + " << blocksToCopy << std::endl;


                    const std::size_t blockStartIndexFirstGPUBlock = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

                    // for current column copy additionalBlocks amount of blocks
                    gpuQueue.submit([&](handler& h) {
                        h.memcpy( &A_gpu[blockStartIndexFirstGPUBlock],&A.matrixData[blockStartIndexFirstGPUBlock], blockSizeBytes * blocksToCopy);
                    });
                }
                gpuQueue.wait();


            } else if (blockCountCPU_new > blockCountCPU) {
                // CPU proportion increased --> move split down
                const int additionalBlocks = blockStartGPU_new - blockStartGPU;

                std::cout << "---- Re-balancing: ----" << std::endl;
                std::cout << "Shifting CPU block count from " << blockCountCPU << " to " << blockCountCPU_new << std::endl;


                // copy part of each column that was computed on the GPU and now has to be computed by the CPU after re-balancing
                for (int c = 0; c < blockCountCPU + k + 1 + (additionalBlocks - 1); ++c) {
                    const int columnsToRight = A.blockCountXY - c;
                    // first block in the column updated by the GPU
                    int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2) ;

                    int blocksToCopy = additionalBlocks;

                    if (c < blockCountCPU + k + 1) {
                        blockID += std::max(A.blockCountXY - c - blockCountGPU, 0);
                    } else {
                        blocksToCopy = additionalBlocks - (c - (blockCountCPU + k + 1) + 1);
                    }

                    std::cout << blockID << " + " << blocksToCopy << std::endl;


                    const std::size_t blockStartIndexFirstGPUBlock = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

                    // for current column copy additionalBlocks amount of blocks
                    gpuQueue.submit([&](handler& h) {
                        h.memcpy(&A.matrixData[blockStartIndexFirstGPUBlock], &A_gpu[blockStartIndexFirstGPUBlock], blockSizeBytes * blocksToCopy);
                    });
                }
                gpuQueue.wait();
            }

            waitAllQueues();

        } else if (blockCountGPU_new != blockCountGPU) {
            // TODO make this cleaner
            std::cout << "Change in block counts smaller than threshold --> no re-balancing: " << blockCountGPU << " --> " << blockCountGPU_new << std::endl;

            // compute new block counts to keep CPU/GPU proportion similar throughout the algorithm
            const int newBlockCountGPU = std::min(std::max(static_cast<int>(std::ceil((A.blockCountXY - 1 - k) * gpuProportion)), minBlockCountGPU), A.blockCountXY - (k + 1));
            const int newBlockCountCPU = std::max(A.blockCountXY - 1 - k - newBlockCountGPU, 0);
            const int newBlockStartGPU = std::max(A.blockCountXY - newBlockCountGPU, k + 1);

            // true if block counts changed and the row that splits the matrix into a CPU and a GPU part has to change
            if (blockCountGPU != newBlockCountGPU && newBlockCountGPU >= minBlockCountGPU && gpuProportion != 1 && gpuProportion != 0) {
                // split has shifted --> communicate the row that was previously held by the GPU and now belongs to the CPU
                shiftSplitRowComm(blockCountATotal, blockSizeBytes, k);
            }

            blockCountGPU = newBlockCountGPU;
            blockCountCPU = newBlockCountCPU;
            blockStartGPU = newBlockStartGPU;
        }
        blockCountGPU = blockCountGPU_new;
        blockCountCPU = blockCountCPU_new;
        blockStartGPU = blockStartGPU_new;

        gpuProportion = gpuProportion_new;
    }
}

void Cholesky::choleskyUpdateCurrentDiagonalBlock(const std::size_t blockSizeBytes, const int k, const int blockID, const std::size_t blockStartIndexDiagBlock) {
    executionTimes.startCholesky = std::chrono::steady_clock::now();
    if ((k < A.blockCountXY - std::max(blockCountGPU, minBlockCountGPU) && gpuProportion != 1) || gpuProportion == 0) {
        // CPU holds A_kk --> use CPU
        MatrixOperations::cholesky(cpuQueue, A.matrixData.data(), blockID, k);
        cpuQueue.wait();

        if (gpuProportion != 0) {
            // copy updated current diagonal block to GPU memory
            gpuQueue.submit([&](handler& h) {
                h.memcpy(&A_gpu[blockStartIndexDiagBlock], &A.matrixData[blockStartIndexDiagBlock], blockSizeBytes);
            }).wait();
        }
    } else {
        // GPU holds A_kk --> use GPU
        MatrixOperations::cholesky_optimizedGPU(gpuQueue, A_gpu, blockID, k);
        gpuQueue.wait();
        if (gpuProportion != 1) {
            // copy updated current diagonal block to CPU memory
            gpuQueue.submit([&](handler& h) {
                h.memcpy(&A.matrixData[blockStartIndexDiagBlock], &A_gpu[blockStartIndexDiagBlock], blockSizeBytes);
            }).wait();
        }
    }
    executionTimes.endCholesky = std::chrono::steady_clock::now();
}

void Cholesky::choleskySolveTriangularSystemColumn(const std::size_t blockSizeBytes, const int k, const int blockID) {
    executionTimes.startTriangularSolve = std::chrono::steady_clock::now();
    if (blockCountCPU > 0) {
        executionTimes.eventCPU_triangularSolve = MatrixMatrixOperations::triangularSolve_optimizedCPU(cpuQueue, A.matrixData.data(), blockID, k, k + 1, blockCountCPU);
    }
    if (blockCountGPU > 0) {
        executionTimes.eventGPU_triangularSolve = MatrixMatrixOperations::triangularSolve_optimizedGPU(gpuQueue, A_gpu, blockID, k, blockStartGPU, blockCountGPU);
    }
    waitAllQueues();

    // if heterogeneous computing is enabled, copy the blocks updated by the CPU to the GPU
    if (gpuProportion != 1 && gpuProportion != 0) {
        executionTimes.startCopy_column = std::chrono::steady_clock::now();
        const std::size_t blockStartIndexFirstCPUSystem = (blockID + 1) * conf::matrixBlockSize * conf::matrixBlockSize;
        // copy updated blocks by CPU to the GPU
        gpuQueue.submit([&](handler& h) {
            h.memcpy(&A_gpu[blockStartIndexFirstCPUSystem], &A.matrixData[blockStartIndexFirstCPUSystem], blockCountCPU * blockSizeBytes);
        }).wait();
        executionTimes.endCopy_column = std::chrono::steady_clock::now();
    }
    executionTimes.endTriangularSolve = std::chrono::steady_clock::now();
}

void Cholesky::choleskyUpdateDiagonal(const int k, const int blockID) {
    executionTimes.startMatrixMatrixDiagonal = std::chrono::steady_clock::now();
    if (blockCountCPU > 0) {
        executionTimes.eventCPU_matrixMatrixDiag = MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedCPU(cpuQueue, A.matrixData.data(), blockID, k, k + 1, blockCountCPU, A.blockCountXY);
    }
    if (blockCountGPU > 0) {
        executionTimes.eventGPU_matrixMatrixDiag = MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedGPU(gpuQueue, A_gpu, blockID, k, blockStartGPU, blockCountGPU, A.blockCountXY);
    }
    waitAllQueues();
    executionTimes.endMatrixMatrixDiagonal = std::chrono::steady_clock::now();
}

void Cholesky::choleskyUpdateLowerBlockTriangle(const int k, const int blockID) {
    executionTimes.startMatrixMatrix = std::chrono::steady_clock::now();
    if (blockCountCPU > 1) {
        executionTimes.eventCPU_matrixMatrix = MatrixMatrixOperations::matrixMatrixStep_optimizedCPU2(cpuQueue, A.matrixData.data(), blockID, k, k + 2, blockCountCPU - 1, A.blockCountXY);
    }
    if (blockCountGPU > 1) {
        executionTimes.eventGPU_matrixMatrix = MatrixMatrixOperations::matrixMatrixStep_optimizedGPU2(gpuQueue, A_gpu, blockID, k, blockStartGPU + offsetMatrixMatrixStepGPU, blockCountGPU - offsetMatrixMatrixStepGPU, A.blockCountXY);
    }
    waitAllQueues();
    executionTimes.endMatrixMatrix = std::chrono::steady_clock::now();
}

void Cholesky::printTimes(const int k) {
    const auto endColumn = std::chrono::steady_clock::now();
    const auto columnTime = std::chrono::duration<double, std::milli>(endColumn - executionTimes.startColumn).count();
    const auto copyTime_row = std::chrono::duration<double, std::milli>(executionTimes.endCopy_row - executionTimes.startCopy_row).count();
    const auto copyTime_column = std::chrono::duration<double, std::milli>(executionTimes.endCopy_column - executionTimes.startCopy_column).count();
    const auto choleskyTime = std::chrono::duration<double, std::milli>(executionTimes.endCholesky - executionTimes.startCholesky).count();
    const auto triangularSolveTime = std::chrono::duration<double, std::milli>(executionTimes.endTriangularSolve - executionTimes.startTriangularSolve).count();
    const auto matrixMatrixDiagonalTime = std::chrono::duration<double, std::milli>(executionTimes.endMatrixMatrixDiagonal - executionTimes.startMatrixMatrixDiagonal).count();
    const auto matrixMatrixTime = std::chrono::duration<double, std::milli>(executionTimes.endMatrixMatrix - executionTimes.startMatrixMatrix).count();

    metricsTracker.shiftTimes.push_back(copyTime_row);
    metricsTracker.choleskyDiagonalBlockTimes.push_back(choleskyTime);
    metricsTracker.triangularSolveTimes_total.push_back(triangularSolveTime);
    metricsTracker.copyTimes.push_back(copyTime_column);
    metricsTracker.matrixMatrixDiagonalTimes_total.push_back(matrixMatrixDiagonalTime);
    metricsTracker.matrixMatrixTimes_total.push_back(matrixMatrixTime);

    metricsTracker.updateMetrics(k, blockCountGPU, blockCountCPU, columnTime, conf::updateInterval);


    std::cout << "Column: " << k << ": " << columnTime << "ms" << std::endl;
    std::cout << "   -- copy row:               " << copyTime_row << "ms" << std::endl;
    std::cout << "   -- cholesky:               " << choleskyTime << "ms" << std::endl;
    std::cout << "   -- triangular solve:       " << triangularSolveTime << "ms" << std::endl;
    if (blockCountCPU > 0) {
        auto triangularSolve_CPU = static_cast<double>(executionTimes.eventCPU_triangularSolve.get_profiling_info<sycl::info::event_profiling::command_end>() - executionTimes.eventCPU_triangularSolve.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6;
        std::cout << "      - CPU:          " << triangularSolve_CPU << "ms" << std::endl;
        metricsTracker.triangularSolveTimes_CPU.push_back(triangularSolve_CPU);
    } else {
        metricsTracker.triangularSolveTimes_CPU.push_back(0);
    }
    if (blockCountGPU > 0) {
        const auto triangularSolve_GPU = static_cast<double>(executionTimes.eventGPU_triangularSolve.get_profiling_info<sycl::info::event_profiling::command_end>() - executionTimes.eventGPU_triangularSolve.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6;
        std::cout << "      - GPU:          " << triangularSolve_GPU << "ms" << std::endl;
        metricsTracker.triangularSolveTimes_GPU.push_back(triangularSolve_GPU);
    } else {
        metricsTracker.triangularSolveTimes_GPU.push_back(0);
    }

    std::cout << "   -- matrix-matrix diagonal: " << matrixMatrixDiagonalTime << "ms" << std::endl;
    if (blockCountCPU > 0) {
        auto matrixMatrixDiagTime_CPU = static_cast<double>(executionTimes.eventCPU_matrixMatrixDiag.get_profiling_info<sycl::info::event_profiling::command_end>() - executionTimes.eventCPU_matrixMatrixDiag.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6;
        std::cout << "      - CPU:          " << matrixMatrixDiagTime_CPU << "ms" << std::endl;
        metricsTracker.matrixMatrixDiagonalTimes_CPU.push_back(matrixMatrixDiagTime_CPU);
    } else {
        metricsTracker.matrixMatrixDiagonalTimes_CPU.push_back(0);
    }
    if (blockCountGPU > 0) {
        const auto matrixMatrixDiagTime_GPU = static_cast<double>(executionTimes.eventGPU_matrixMatrixDiag.get_profiling_info<sycl::info::event_profiling::command_end>() - executionTimes.eventGPU_matrixMatrixDiag.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6;
        std::cout << "      - GPU:          " << matrixMatrixDiagTime_GPU << "ms" << std::endl;
        metricsTracker.matrixMatrixDiagonalTimes_GPU.push_back(matrixMatrixDiagTime_GPU);
    } else {
        metricsTracker.matrixMatrixDiagonalTimes_GPU.push_back(0);
    }

    std::cout << "   -- matrix-matrix:          " << matrixMatrixTime << "ms" << std::endl;
    if (blockCountCPU > 0) {
        auto matrixMatrixTime_CPU = static_cast<double>(executionTimes.eventCPU_matrixMatrix.get_profiling_info<sycl::info::event_profiling::command_end>() - executionTimes.eventCPU_matrixMatrix.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6;
        std::cout << "      - CPU:          " << matrixMatrixTime_CPU << "ms" << std::endl;
        metricsTracker.matrixMatrixTimes_CPU.push_back(matrixMatrixTime_CPU);
    } else {
        metricsTracker.matrixMatrixTimes_CPU.push_back(0);
    }
    if (blockCountGPU > 0) {
        const auto matrixMatrixTime_GPU = static_cast<double>(executionTimes.eventGPU_matrixMatrix.get_profiling_info<sycl::info::event_profiling::command_end>() - executionTimes.eventGPU_matrixMatrix.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6;
        std::cout << "      - GPU:          " << matrixMatrixTime_GPU << "ms" << std::endl;
        metricsTracker.matrixMatrixTimes_GPU.push_back(matrixMatrixTime_GPU);
    } else {
        metricsTracker.matrixMatrixTimes_GPU.push_back(0);
    }

    std::cout << std::endl;
    std::cout << std::endl;
}

void Cholesky::copyResultFromGPU(const int blockCountATotal, const std::size_t blockSizeBytes) {
    executionTimes.startResultCopyGPU = std::chrono::steady_clock::now();
    if (gpuProportion != 1 && gpuProportion != 0) {
        // Case heterogeneous: copy parts of the matrix that were computed by the GPU to the CPU

        // copy part of each column that was computed on the GPU to the CPU
        for (int k = 0; k < A.blockCountXY; ++k) {
            const int columnsToRight = A.blockCountXY - k;
            // first block in the column updated by the GPU
            const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2) + std::max(A.blockCountXY - k - minBlockCountGPU, 0);
            const std::size_t blockStartIndexFirstGPUBlock = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

            int blockCountGPUinColumn;
            if (k <= minBlockCountGPU) {
                blockCountGPUinColumn = minBlockCountGPU;
            } else {
                blockCountGPUinColumn = A.blockCountXY - k;
            }

            gpuQueue.submit([&](handler& h) {
                h.memcpy(&A.matrixData[blockStartIndexFirstGPUBlock], &A_gpu[blockStartIndexFirstGPUBlock], blockCountGPUinColumn * blockSizeBytes);
            });
        }
        gpuQueue.wait();
    } else if (gpuProportion == 1) {
        // Case GPU-only: copy complete matrix to the CPU

        gpuQueue.submit([&](handler& h) {
            h.memcpy(A.matrixData.data(), A_gpu, A.matrixData.size() * sizeof(conf::fp_type));
        }).wait();
    }
    executionTimes.endResultCopyGPU = std::chrono::steady_clock::now();
}

void Cholesky::printFinalTimes() {
    const auto memoryInitTime = std::chrono::duration<double, std::milli>(executionTimes.endMemoryInitGPU - executionTimes.startMemoryInitGPU).count();
    const auto resultCopyTime = std::chrono::duration<double, std::milli>(executionTimes.endResultCopyGPU - executionTimes.startResultCopyGPU).count();
    const auto totalTime = std::chrono::duration<double, std::milli>(executionTimes.end - executionTimes.start).count();

    std::cout << "Memory init: " << memoryInitTime << "ms" << std::endl;
    std::cout << "Result copy: " << resultCopyTime << std::endl;
    std::cout << "Total time:  " << totalTime << "ms" << std::endl;
}

void Cholesky::solve_heterogeneous() {
    executionTimes.start = std::chrono::steady_clock::now();

    gpuProportion = conf::initialProportionGPU;

    if (gpuProportion != 0) {
        minBlockCountGPU = conf::minBlockCountCholesky;
    } else {
        minBlockCountGPU = 0;
    }

    // helper variables for various calculations
    const int blockCountATotal = (A.blockCountXY * (A.blockCountXY + 1) / 2);
    const std::size_t blockSizeBytes = conf::matrixBlockSize * conf::matrixBlockSize * sizeof(conf::fp_type);

    // initialize GPU memory if the GPU should be used
    executionTimes.startMemoryInitGPU = std::chrono::steady_clock::now();
    executionTimes.endMemoryInitGPU = executionTimes.startMemoryInitGPU;
    if (gpuProportion != 0) {
        initGPUMemory();
    }

    // Calculate initial block-count values for CPU and GPU. These values correspond to the number of row-blocks
    const int initialBlockCountGPU = std::ceil((A.blockCountXY - 1) * gpuProportion);
    const int initialBlockCountCPU = A.blockCountXY - 1 - initialBlockCountGPU;
    const int initialBlockStartGPU = initialBlockCountCPU + 1;
    blockCountGPU = initialBlockCountGPU;
    blockCountCPU = initialBlockCountCPU;
    blockStartGPU = initialBlockStartGPU;
    metricsTracker.startTracking();


    // begin with tiled Cholesky decomposition using right-looking algorithm
    for (int k = 0; k < A.blockCountXY; ++k) {
        initExecutionTimes();

        // check if row that splits GPU/CPU part of the matrix has to change and apply the change if necessary
        shiftSplit(blockCountATotal, blockSizeBytes, k);

        // change offset for matrix-matrix step from 0 to 1 depending on how far we are into the computation
        if (blockStartGPU <= k + 1) {
            offsetMatrixMatrixStepGPU = 1;
        }

        // ID and start index of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - k;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);
        const std::size_t blockStartIndexDiagBlock = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

        // perform Cholesky decomposition on diagonal block A_kk
        choleskyUpdateCurrentDiagonalBlock(blockSizeBytes, k, blockID, blockStartIndexDiagBlock);

        // solve triangular system for current column k below the diagonal
        choleskySolveTriangularSystemColumn(blockSizeBytes, k, blockID);

        // update the blocks on the diagonal below the current diagonal block
        choleskyUpdateDiagonal(k, blockID);

        // update the blocks in the lower triangle below the current diagonal block
        choleskyUpdateLowerBlockTriangle(k, blockID);

        // time measurement and output
        printTimes(k);
    }

    metricsTracker.endTracking();

    // copies all values that have been computed on GPU and are not yet in CPU memory
    copyResultFromGPU(blockCountATotal, blockSizeBytes);

    executionTimes.end = std::chrono::steady_clock::now();
    printFinalTimes();

    std::string timeString = UtilityFunctions::getTimeString();
    std::string filePath = conf::outputPath + "/" + timeString;
    std::filesystem::create_directories(filePath);
    metricsTracker.writeJSON(filePath);

    if (conf::writeResult) {
        MatrixParser::writeFullMatrix("./A_chol_result", A);
    }
}

void Cholesky::solve() {
    bool useGPU = true;
    const auto start = std::chrono::steady_clock::now();

    // init GPU data structure for matrix A on which the Cholesky decomposition should be performed
    if (useGPU) {
        A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);
        // Copy the matrix A to the GPU
        gpuQueue.submit([&](handler& h) {
            h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
        }).wait();
    }

    const int blockCountATotal = (A.blockCountXY * (A.blockCountXY + 1) / 2);


    // begin with tiled Cholesky decomposition using right-looking algorithm
    for (int k = 0; k < A.blockCountXY; ++k) {
        auto startColumn = std::chrono::steady_clock::now();

        auto startCholesky = std::chrono::steady_clock::now();
        auto endCholesky = startCholesky;

        auto startTriangularSolve = std::chrono::steady_clock::now();
        auto endTriangularSolve = startTriangularSolve;

        auto startMatrixMatrixDiagonal = std::chrono::steady_clock::now();
        auto endMatrixMatrixDiagonal = startMatrixMatrixDiagonal;

        auto startMatrixMatrix = std::chrono::steady_clock::now();
        auto endMatrixMatrix = startMatrixMatrix;

        // ID of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - k;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);

        // perform Cholesky decomposition on diagonal block A_kk
        startCholesky = std::chrono::steady_clock::now();
        if (useGPU) {
            MatrixOperations::cholesky_optimizedGPU(gpuQueue, A_gpu, blockID, k);
            gpuQueue.wait();
        } else {
            MatrixOperations::cholesky(cpuQueue, A.matrixData.data(), blockID, k);
            cpuQueue.wait();
        }

        endCholesky = std::chrono::steady_clock::now();

        if (k < A.blockCountXY - 1) {
            // solve triangular system for current column k below the diagonal
            startTriangularSolve = std::chrono::steady_clock::now();
            if (useGPU) {
                MatrixMatrixOperations::triangularSolve_optimizedGPU(gpuQueue, A_gpu, blockID, k, k + 1,
                                                                     A.blockCountXY - (k + 1));
                gpuQueue.wait();
            } else {
                MatrixMatrixOperations::triangularSolve_optimizedCPU(cpuQueue, A.matrixData.data(), blockID, k, k + 1,
                                                                     A.blockCountXY - (k + 1));
                cpuQueue.wait();
            }

            endTriangularSolve = std::chrono::steady_clock::now();

            // update the blocks on the diagonal below the current diagonal block
            startMatrixMatrixDiagonal = std::chrono::steady_clock::now();
            if (useGPU) {
                MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedGPU(
                    gpuQueue, A_gpu, blockID, k, k + 1, A.blockCountXY - (k + 1), A.blockCountXY);
                gpuQueue.wait();
            } else {
                MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedCPU(
                    cpuQueue, A.matrixData.data(), blockID, k, k + 1,
                    A.blockCountXY - (k + 1), A.blockCountXY);
                cpuQueue.wait();
            }

            endMatrixMatrixDiagonal = std::chrono::steady_clock::now();

            if (k < A.blockCountXY - 2) {
                // update the blocks int the lower triangle below the current diagonal block
                startMatrixMatrix = std::chrono::steady_clock::now();
                if (useGPU) {
                    MatrixMatrixOperations::matrixMatrixStep_optimizedGPU3(
                        gpuQueue, A_gpu, blockID, k, k + 2, A.blockCountXY - (k + 2), A.blockCountXY);
                    gpuQueue.wait();
                } else {
                    MatrixMatrixOperations::matrixMatrixStep_optimizedCPU2(
                        cpuQueue, A.matrixData.data(), blockID, k, k + 2,
                        A.blockCountXY - (k + 2), A.blockCountXY);
                    cpuQueue.wait();
                }
                endMatrixMatrix = std::chrono::steady_clock::now();
            }
        }

        auto endColumn = std::chrono::steady_clock::now();
        auto columnTime = std::chrono::duration<double, std::milli>(endColumn - startColumn).count();

        auto choleskyTime = std::chrono::duration<double, std::milli>(endCholesky - startCholesky).count();
        auto triangularSolveTime = std::chrono::duration<double, std::milli>(endTriangularSolve - startTriangularSolve).
            count();
        auto matrixMatrixDiagonalTime = std::chrono::duration<double, std::milli>(
            endMatrixMatrixDiagonal - startMatrixMatrixDiagonal).count();
        auto matrixMatrixTime = std::chrono::duration<double, std::milli>(endMatrixMatrix - startMatrixMatrix).count();

        std::cout << "Column: " << k << ": " << columnTime << "ms" << std::endl;
        std::cout << "   -- cholesky:               " << choleskyTime << "ms" << std::endl;
        std::cout << "   -- triangular solve:       " << triangularSolveTime << "ms" << std::endl;
        std::cout << "   -- matrix-matrix diagonal: " << matrixMatrixDiagonalTime << "ms" << std::endl;
        std::cout << "   -- matrix-matrix:          " << matrixMatrixTime << "ms" << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    if (useGPU) {
        gpuQueue.submit([&](handler& h) {
            h.memcpy(A.matrixData.data(), A_gpu, A.matrixData.size() * sizeof(conf::fp_type));
        }).wait();
    }


    const auto end = std::chrono::steady_clock::now();
    const auto totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << totalTime << "ms" << std::endl;

    MatrixParser::writeFullMatrix("./A_chol_result", A);
}

void Cholesky::waitAllQueues() {
    if (blockCountGPU != 0) {
        gpuQueue.wait();
    }
    if (blockCountCPU != 0) {
        cpuQueue.wait();
    }
}
