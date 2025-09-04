#ifndef TRIANGULARSYSTEMSOLVER_HPP
#define TRIANGULARSYSTEMSOLVER_HPP

#include "LoadBalancer.hpp"
#include "SymmetricMatrix.hpp"
#include "MetricsTracker.hpp"
#include "RightHandSide.hpp"

using namespace sycl;

class TriangularSystemSolver {
public:
    TriangularSystemSolver(SymmetricMatrix& A,  conf::fp_type* A_gpu, RightHandSide& b,   queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A;
    RightHandSide& b;

    // GPU data structure
    conf::fp_type* A_gpu;
    conf::fp_type* b_gpu;


    queue& cpuQueue;
    queue& gpuQueue;

    std::shared_ptr<LoadBalancer> loadBalancer;
    MetricsTracker metricsTracker;

    double solve();

};



#endif //TRIANGULARSYSTEMSOLVER_HPP
