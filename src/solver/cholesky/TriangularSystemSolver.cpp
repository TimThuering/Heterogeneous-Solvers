#include "TriangularSystemSolver.hpp"

#include "MatrixVectorOperations.hpp"

TriangularSystemSolver::TriangularSystemSolver(SymmetricMatrix& A, RightHandSide& b, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer) :
    A(A),
    b(b),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue),
    loadBalancer(std::move(loadBalancer)) {
}

void TriangularSystemSolver::solve() {
    const auto start = std::chrono::steady_clock::now();

    // helper variables for various calculations
    const int blockCountATotal = (A.blockCountXY * (A.blockCountXY + 1) / 2);

    // for each column in lower triangular matrix
    for (int j = 0; j < A.blockCountXY; ++j) {
        // ID and start index of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - j;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);

        // Solve triangular system for diagonal block: b_j = Solve(A_jj, b_j)
        MatrixVectorOperations::triangularSolveBlockVector(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), j, blockID, false);
        cpuQueue.wait();

        // Update column below diagonal block
        MatrixVectorOperations::matrixVectorColumnUpdate(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), j + 1, A.blockCountXY - (j + 1), j, blockID, A.blockCountXY, false);
        cpuQueue.wait();
    }

    // for each column in upper triangular matrix --> each row with transposed blocks in the lower triangular matrix
    for (int j = A.blockCountXY - 1; j >= 0; --j) {
        // ID and start index of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - j;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);

        // Solve triangular system for diagonal block: b_j = Solve(A_jj, b_j)
        MatrixVectorOperations::triangularSolveBlockVector(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), j, blockID, true);
        cpuQueue.wait();

        // Update column below diagonal block
        MatrixVectorOperations::matrixVectorColumnUpdate(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), 0, j, j, blockID, A.blockCountXY, true);
        cpuQueue.wait();
    }

    const auto end = std::chrono::steady_clock::now();
    const auto solveTime = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Time to solve the triangular system: " << solveTime << "ms" << std::endl;
}
