#ifndef TRIANGULARSYSTEMSOLVER_HPP
#define TRIANGULARSYSTEMSOLVER_HPP

#include "LoadBalancer.hpp"
#include "SymmetricMatrix.hpp"
#include "MetricsTracker.hpp"

using namespace sycl;

class TriangularSystemSolver {
public:
    TriangularSystemSolver(SymmetricMatrix& A, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A;

    queue& cpuQueue;
    queue& gpuQueue;

    std::shared_ptr<LoadBalancer> loadBalancer;
    MetricsTracker metricsTracker;

};



#endif //TRIANGULARSYSTEMSOLVER_HPP
