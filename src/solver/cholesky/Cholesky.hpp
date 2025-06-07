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

    // variables
    int blockCountGPU;
    int blockCountCPU;
    int blockStartGPU;
    int offsetMatrixMatrixStepGPU = 0;

    void waitAllQueues();



};



#endif //CHOLESKY_HPP
