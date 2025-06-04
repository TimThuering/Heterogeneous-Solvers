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

    void solve();

private:
    conf::fp_type* A_gpu;


};



#endif //CHOLESKY_HPP
