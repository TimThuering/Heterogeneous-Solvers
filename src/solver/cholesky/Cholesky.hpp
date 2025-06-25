#ifndef CHOLESKY_HPP
#define CHOLESKY_HPP

#include <sycl/sycl.hpp>

#include "LoadBalancer.hpp"
#include "MetricsTracker.hpp"
#include "SymmetricMatrix.hpp"

using namespace sycl;

class Cholesky {
public:
    Cholesky(SymmetricMatrix& A, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A;

    queue& cpuQueue;
    queue& gpuQueue;

    std::shared_ptr<LoadBalancer> loadBalancer;
    MetricsTracker metricsTracker;

    double gpuProportion;


    void solve_heterogeneous();

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

        std::chrono::time_point<std::chrono::steady_clock> startCopy_row;
        std::chrono::time_point<std::chrono::steady_clock> endCopy_row;

        std::chrono::time_point<std::chrono::steady_clock> startCopy_column;
        std::chrono::time_point<std::chrono::steady_clock> endCopy_column;

        std::chrono::time_point<std::chrono::steady_clock> startTriangularSolve;
        std::chrono::time_point<std::chrono::steady_clock> endTriangularSolve;

        std::chrono::time_point<std::chrono::steady_clock> startMatrixMatrixDiagonal;
        std::chrono::time_point<std::chrono::steady_clock> endMatrixMatrixDiagonal;

        std::chrono::time_point<std::chrono::steady_clock> startMatrixMatrix;
        std::chrono::time_point<std::chrono::steady_clock> endMatrixMatrix;

        std::chrono::time_point<std::chrono::steady_clock> startResultCopyGPU;
        std::chrono::time_point<std::chrono::steady_clock> endResultCopyGPU;

        sycl::event eventCPU_matrixMatrix;
        sycl::event eventGPU_matrixMatrix;

        sycl::event eventCPU_matrixMatrixDiag;
        sycl::event eventGPU_matrixMatrixDiag;

        sycl::event eventCPU_triangularSolve;
        sycl::event eventGPU_triangularSolve;
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
    void shiftSplit(int blockCountATotal, std::size_t blockSizeBytes, int k);
    void choleskyUpdateCurrentDiagonalBlock(std::size_t blockSizeBytes, int k, int blockID, std::size_t blockStartIndexDiagBlock);
    void choleskySolveTriangularSystemColumn(std::size_t blockSizeBytes, int k, int blockID);
    void choleskyUpdateDiagonal(int k, int blockID);
    void choleskyUpdateLowerBlockTriangle(int k, int blockID);
    void printTimes(int k);
    void copyResultFromGPU(int blockCountATotal, std::size_t blockSizeBytes);
    void printFinalTimes();
};


#endif //CHOLESKY_HPP
