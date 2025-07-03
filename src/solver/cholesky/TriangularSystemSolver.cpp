#include "TriangularSystemSolver.hpp"

TriangularSystemSolver::TriangularSystemSolver(SymmetricMatrix& A, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer) :
    A(A),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue),
    loadBalancer(std::move(loadBalancer)) {
}