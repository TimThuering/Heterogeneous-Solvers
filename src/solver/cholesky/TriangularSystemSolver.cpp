#include "TriangularSystemSolver.hpp"

#include "MatrixVectorOperations.hpp"
#include "UtilityFunctions.hpp"

TriangularSystemSolver::TriangularSystemSolver(SymmetricMatrix& A, conf::fp_type* A_gpu, RightHandSide& b, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer) :
    A(A),
    b(b),
    A_gpu(A_gpu),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue),
    loadBalancer(std::move(loadBalancer)) {
}

double TriangularSystemSolver::solve() {
    const auto start = std::chrono::steady_clock::now();

    bool useGPU = conf::initialProportionGPU == 1;

    if (useGPU) {
        b_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);
        gpuQueue.submit([&](handler& h) {
            h.memcpy(b_gpu, b.rightHandSideData.data(), b.rightHandSideData.size() * sizeof(conf::fp_type));
        }).wait();
    }

    // helper variables for various calculations
    const int blockCountATotal = (A.blockCountXY * (A.blockCountXY + 1) / 2);

    // for each column in lower triangular matrix
    for (int j = 0; j < A.blockCountXY; ++j) {
        // ID and start index of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - j;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);

        // Solve triangular system for diagonal block: b_j = Solve(A_jj, b_j)
        if (useGPU) {
            MatrixVectorOperations::triangularSolveBlockVector(gpuQueue, A_gpu, b_gpu, j, blockID, false);
            gpuQueue.wait();
        } else {
            MatrixVectorOperations::triangularSolveBlockVector(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), j, blockID, false);
            cpuQueue.wait();
        }

        if (j < A.blockCountXY - 1) {
            // Update column below diagonal block
            if (useGPU) {
                MatrixVectorOperations::matrixVectorColumnUpdate(gpuQueue, A_gpu, b_gpu, j + 1, A.blockCountXY - (j + 1), j, blockID, A.blockCountXY, false);
                gpuQueue.wait();
            } else {
                MatrixVectorOperations::matrixVectorColumnUpdate(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), j + 1, A.blockCountXY - (j + 1), j, blockID, A.blockCountXY, false);
                cpuQueue.wait();
            }
        }
    }

    // for each column in upper triangular matrix --> each row with transposed blocks in the lower triangular matrix
    for (int j = A.blockCountXY - 1; j >= 0; --j) {
        // ID and start index of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - j;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);

        // Solve triangular system for diagonal block: b_j = Solve(A_jj, b_j)
        if (useGPU) {
            MatrixVectorOperations::triangularSolveBlockVector(gpuQueue, A_gpu, b_gpu, j, blockID, true);
            gpuQueue.wait();
        } else {
            MatrixVectorOperations::triangularSolveBlockVector(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), j, blockID, true);
            cpuQueue.wait();
        }

        if (j > 0) {
            // Update column below diagonal block
            if (useGPU) {
                MatrixVectorOperations::matrixVectorColumnUpdate(gpuQueue, A_gpu, b_gpu, 0, j, j, blockID, A.blockCountXY, true);
                gpuQueue.wait();
            } else {
                MatrixVectorOperations::matrixVectorColumnUpdate(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), 0, j, j, blockID, A.blockCountXY, true);
                cpuQueue.wait();
            }
        }
    }

    if (useGPU) {
        // copy data back to the CPU
        gpuQueue.submit([&](handler& h) {
            h.memcpy(b.rightHandSideData.data(), b_gpu, b.rightHandSideData.size() * sizeof(conf::fp_type));
        }).wait();
    }

    const auto end = std::chrono::steady_clock::now();
    const double solveTime = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Time to solve the triangular system: " << solveTime << "ms" << std::endl;

    if (conf::writeResult) {
        UtilityFunctions::writeResult(".", b.rightHandSideData);
    }

    if (useGPU) {
        sycl::free(b_gpu, gpuQueue);
    }

    return solveTime;
}
