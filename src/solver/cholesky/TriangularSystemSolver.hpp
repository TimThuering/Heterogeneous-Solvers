#ifndef TRIANGULARSYSTEMSOLVER_HPP
#define TRIANGULARSYSTEMSOLVER_HPP

#include "LoadBalancer.hpp"
#include "SymmetricMatrix.hpp"
#include "MetricsTracker.hpp"
#include "RightHandSide.hpp"

using namespace sycl;

class TriangularSystemSolver {
public:
    TriangularSystemSolver(SymmetricMatrix& A, RightHandSide& b,  queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A;
    RightHandSide& b;


    queue& cpuQueue;
    queue& gpuQueue;

    std::shared_ptr<LoadBalancer> loadBalancer;
    MetricsTracker metricsTracker;

    void solve();

};



#endif //TRIANGULARSYSTEMSOLVER_HPP
