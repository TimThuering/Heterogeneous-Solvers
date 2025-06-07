#include "Cholesky.hpp"

#include "MatrixMatrixOperations.hpp"
#include "MatrixOperations.hpp"
#include "MatrixParser.hpp"

Cholesky::Cholesky(SymmetricMatrix& A, queue& cpuQueue, queue& gpuQueue):
    A(A),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue) {
}

void Cholesky::solve_heterogeneous() {
    const double gpuProportion = 0.6;
    const auto start = std::chrono::steady_clock::now();

    // init GPU data structure for matrix A on which the Cholesky decomposition should be performed
    A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);
    // Copy the matrix A to the GPU
    gpuQueue.submit([&](handler& h) {
        h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
    }).wait();

    const int blockCountATotal = (A.blockCountXY * (A.blockCountXY + 1) / 2);

    const std::size_t blockSizeBytes = conf::matrixBlockSize * conf::matrixBlockSize * sizeof(conf::fp_type);

    // Calculate initial block-count values for CPU and GPU. These values correspond to the number of row-blocks
    const int initialBlockCountGPU = std::ceil((A.blockCountXY - 1) * gpuProportion);
    const int initialBlockCountCPU = A.blockCountXY - 1 - initialBlockCountGPU;
    const int initialBlockStartGPU = initialBlockCountCPU + 1;

    blockCountGPU = initialBlockCountGPU;
    blockCountCPU = initialBlockCountCPU;
    blockStartGPU = initialBlockStartGPU;


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

        // reduce CPU blockCount
        blockCountCPU = std::max(A.blockCountXY - 1 - k - initialBlockCountGPU, 0);
        blockCountGPU = std::min(blockCountGPU, A.blockCountXY - (k + 1));
        blockStartGPU = std::max(blockStartGPU, k + 1);
        // change offset for matrix-matrix step from 0 to 1 depending on how far we are into the computation
        if (blockStartGPU != initialBlockStartGPU) {
            offsetMatrixMatrixStepGPU = 1;
        }


        // ID of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - k;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);
        const std::size_t blockStartIndexDiagBlock = blockID * conf::matrixBlockSize * conf::matrixBlockSize;


        // perform Cholesky decomposition on diagonal block A_kk
        startCholesky = std::chrono::steady_clock::now();
        if (k < initialBlockStartGPU) {
            // CPU holds A_kk --> use CPU
            MatrixOperations::cholesky(cpuQueue, A.matrixData.data(), blockID, k);
            cpuQueue.wait();
            // copy updated current diagonal block to GPU memory
            gpuQueue.submit([&](handler& h) {
                h.memcpy(&A_gpu[blockStartIndexDiagBlock], &A.matrixData[blockStartIndexDiagBlock], blockSizeBytes);
            }).wait();
        } else {
            // GPU holds A_kk --> use GPU
            MatrixOperations::cholesky_optimizedGPU(gpuQueue, A_gpu, blockID, k);
            gpuQueue.wait();
            // copy updated current diagonal block to CPU memory
            gpuQueue.submit([&](handler& h) {
                h.memcpy(&A.matrixData[blockStartIndexDiagBlock], &A_gpu[blockStartIndexDiagBlock], blockSizeBytes);
            }).wait();
        }
        endCholesky = std::chrono::steady_clock::now();

        // solve triangular system for current column k below the diagonal
        startTriangularSolve = std::chrono::steady_clock::now();
        if (blockCountCPU > 0 && k < initialBlockCountCPU) {
            MatrixMatrixOperations::triangularSolve_optimizedCPU(cpuQueue, A.matrixData.data(), blockID, k, k + 1, blockCountCPU);
        }
        if (blockCountGPU > 0 && k < A.blockCountXY - 1) {
            MatrixMatrixOperations::triangularSolve_optimizedGPU(gpuQueue, A_gpu, blockID, k, blockStartGPU, blockCountGPU);
        }
        waitAllQueues();
        const std::size_t blockStartIndexFirstCPUSystem = (blockID + 1) * conf::matrixBlockSize * conf::matrixBlockSize;
        // copy updated blocks by CPU to the GPU
        gpuQueue.submit([&](handler& h) {
            h.memcpy(&A_gpu[blockStartIndexFirstCPUSystem], &A.matrixData[blockStartIndexFirstCPUSystem], blockCountCPU * blockSizeBytes);
        }).wait();
        endTriangularSolve = std::chrono::steady_clock::now();


        // update the blocks on the diagonal below the current diagonal block
        startMatrixMatrixDiagonal = std::chrono::steady_clock::now();
        if (blockCountCPU > 0 && k < initialBlockCountCPU) {
            MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedCPU(cpuQueue, A.matrixData.data(), blockID, k, k + 1, blockCountCPU, A.blockCountXY);
        }
        if (blockCountGPU > 0 && k < A.blockCountXY - 1) {
            MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedGPU(gpuQueue, A_gpu, blockID, k, blockStartGPU, blockCountGPU, A.blockCountXY);
        }
        waitAllQueues();
        endMatrixMatrixDiagonal = std::chrono::steady_clock::now();


        // update the blocks int the lower triangle below the current diagonal block
        startMatrixMatrix = std::chrono::steady_clock::now();
        if (blockCountCPU > 0 && k < initialBlockCountCPU - 1) {
            MatrixMatrixOperations::matrixMatrixStep_optimizedCPU(cpuQueue, A.matrixData.data(), blockID, k, k + 2, blockCountCPU - 1, A.blockCountXY);
        }
        if (blockCountGPU > 0 && k < A.blockCountXY - 2) {
            MatrixMatrixOperations::matrixMatrixStep_optimizedGPU(gpuQueue, A_gpu, blockID, k, blockStartGPU + offsetMatrixMatrixStepGPU, blockCountGPU - offsetMatrixMatrixStepGPU, A.blockCountXY);
        }
        waitAllQueues();
        endMatrixMatrix = std::chrono::steady_clock::now();


        // time measurement and output
        auto endColumn = std::chrono::steady_clock::now();
        auto columnTime = std::chrono::duration<double, std::milli>(endColumn - startColumn).count();

        auto choleskyTime = std::chrono::duration<double, std::milli>(endCholesky - startCholesky).count();
        auto triangularSolveTime = std::chrono::duration<double, std::milli>(endTriangularSolve - startTriangularSolve).count();
        auto matrixMatrixDiagonalTime = std::chrono::duration<double, std::milli>(endMatrixMatrixDiagonal - startMatrixMatrixDiagonal).count();
        auto matrixMatrixTime = std::chrono::duration<double, std::milli>(endMatrixMatrix - startMatrixMatrix).count();

        std::cout << "Column: " << k << ": " << columnTime << "ms" << std::endl;
        std::cout << "   -- cholesky:               " << choleskyTime << "ms" << std::endl;
        std::cout << "   -- triangular solve:       " << triangularSolveTime << "ms" << std::endl;
        std::cout << "   -- matrix-matrix diagonal: " << matrixMatrixDiagonalTime << "ms" << std::endl;
        std::cout << "   -- matrix-matrix:          " << matrixMatrixTime << "ms" << std::endl;
        // std::cout << "   -- block count CPU:        " << blockCountCPU << std::endl;
        // std::cout << "   -- block count GPU:        " << blockCountGPU << std::endl;
        // std::cout << "   -- block start GPU:        " << blockStartGPU << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // copy part of each column that was computed on the GPU to the CPU
    for (int k = 0; k < A.blockCountXY; ++k) {
        const int columnsToRight = A.blockCountXY - k;
        // first block in the column updated by the GPU
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2) + std::max(A.blockCountXY - k - initialBlockCountGPU, 0);
        const std::size_t blockStartIndexFirstGPUBlock = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

        int blockCountGPUinColumn;
        if (k <= initialBlockCountGPU) {
            blockCountGPUinColumn = initialBlockCountGPU;
        } else {
            blockCountGPUinColumn = A.blockCountXY - k;
        }

        gpuQueue.submit([&](handler& h) {
            h.memcpy(&A.matrixData[blockStartIndexFirstGPUBlock], &A_gpu[blockStartIndexFirstGPUBlock], blockCountGPUinColumn * blockSizeBytes);
        });
    }
    gpuQueue.wait();


    const auto end = std::chrono::steady_clock::now();
    const auto totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << totalTime << "ms" << std::endl;

    MatrixParser::writeFullMatrix("./A_chol_result", A);
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
                    MatrixMatrixOperations::matrixMatrixStep_optimizedGPU(
                        gpuQueue, A_gpu, blockID, k, k + 2, A.blockCountXY - (k + 2), A.blockCountXY);
                    gpuQueue.wait();
                } else {
                    MatrixMatrixOperations::matrixMatrixStep_optimizedCPU(
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
