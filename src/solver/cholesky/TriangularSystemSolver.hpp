#ifndef TRIANGULARSYSTEMSOLVER_HPP
#define TRIANGULARSYSTEMSOLVER_HPP

#include "LoadBalancer.hpp"
#include "SymmetricMatrix.hpp"
#include "MetricsTracker.hpp"
#include "RightHandSide.hpp"

using namespace sycl;

/**
 * This class contains a solver for the resulting triangular systems that emerge after the Cholesky decomposition has been perfomred.
 * The systems are solved using a forward and back substitution.
 */
class TriangularSystemSolver {
public:
    TriangularSystemSolver(SymmetricMatrix& A,  conf::fp_type* A_gpu, RightHandSide& b,   queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A; /// SPD matrix A
    RightHandSide& b; /// GPU data structure for A

    // GPU data structure
    conf::fp_type* A_gpu; /// GPU data structure for A
    conf::fp_type* b_gpu; /// GPU data structure for b


    queue& cpuQueue; /// SYCL queue for the CPU device
    queue& gpuQueue; /// SYCL queue for the GPU device

    std::shared_ptr<LoadBalancer> loadBalancer; /// load balancer object
    MetricsTracker metricsTracker; /// metrics tracker object

    double solve();

};



#endif //TRIANGULARSYSTEMSOLVER_HPP
