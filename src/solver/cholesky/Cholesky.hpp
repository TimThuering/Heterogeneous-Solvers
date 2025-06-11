#ifndef CHOLESKY_HPP
#define CHOLESKY_HPP

#include <sycl/sycl.hpp>

#include "SymmetricMatrix.hpp"

using namespace sycl;

class Cholesky {
public:
    Cholesky(SymmetricMatrix& A,queue& cpuQueue, queue& gpuQueue);

    SymmetricMatrix& A;

    queue& cpuQueue;
    queue& gpuQueue;

    void solve_heterogeneous();

    void solve();

private:
    // GPU data structure
    conf::fp_type* A_gpu;

    struct executionTimes {
        std::chrono::time_point<std::chrono::steady_clock> start;
        std::chrono::time_point<std::chrono::steady_clock> end;


        std::chrono::time_point<std::chrono::steady_clock> startMemoryInitGPU;
        std::chrono::time_point<std::chrono::steady_clock> endMemoryInitGPU;

        std::chrono::time_point<std::chrono::steady_clock> startColumn;

        std::chrono::time_point<std::chrono::steady_clock> startCholesky;
        std::chrono::time_point<std::chrono::steady_clock> endCholesky;

        std::chrono::time_point<std::chrono::steady_clock> startCopy;
        std::chrono::time_point<std::chrono::steady_clock> endCopy;

        std::chrono::time_point<std::chrono::steady_clock> startTriangularSolve;
        std::chrono::time_point<std::chrono::steady_clock> endTriangularSolve;

        std::chrono::time_point<std::chrono::steady_clock> startMatrixMatrixDiagonal;
        std::chrono::time_point<std::chrono::steady_clock> endMatrixMatrixDiagonal;

        std::chrono::time_point<std::chrono::steady_clock> startMatrixMatrix;
        std::chrono::time_point<std::chrono::steady_clock> endMatrixMatrix;

        std::chrono::time_point<std::chrono::steady_clock> startResultCopyGPU;
        std::chrono::time_point<std::chrono::steady_clock> endResultCopyGPU;

        sycl::event eventCPU;
        sycl::event eventGPU;

    } executionTimes;

    // variables
    int blockCountGPU;
    int blockCountCPU;
    int blockStartGPU;
    int offsetMatrixMatrixStepGPU = 0;
    int minBlockCountGPU;


    void waitAllQueues();

    void initGPUMemory();
    void initExecutionTimes();
    void shiftSplitRowComm(int blockCountATotal, std::size_t blockSizeBytes, int k);
    void shiftSplit(double gpuProportion, int blockCountATotal, std::size_t blockSizeBytes, int k);
    void choleskyUpdateCurrentDiagonalBlock(double gpuProportion, std::size_t blockSizeBytes, int k, int blockID, std::size_t blockStartIndexDiagBlock);
    void choleskySolveTriangularSystemColumn(double gpuProportion, std::size_t blockSizeBytes, int k, int blockID);
    void choleskyUpdateDiagonal(int k, int blockID);
    void choleskyUpdateLowerBlockTriangle(double gpuProportion, int k, int blockID);
    void printTimes(int k) const;
    void copyResultFromGPU(double gpuProportion, int blockCountATotal, std::size_t blockSizeBytes);
    void printFinalTimes() const;
};



#endif //CHOLESKY_HPP
