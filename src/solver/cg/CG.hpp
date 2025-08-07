#ifndef CG_HPP
#define CG_HPP

#include <string>
#include <sycl/sycl.hpp>

#include "SymmetricMatrix.hpp"
#include "RightHandSide.hpp"
#include "Configuration.hpp"
#include "LoadBalancer.hpp"
#include "MetricsTracker.hpp"

using namespace sycl;

class CG {
public:
    CG(SymmetricMatrix& A,RightHandSide& b, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A;
    RightHandSide& b;

    std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>> x;

    queue& cpuQueue;
    queue& gpuQueue;

    std::shared_ptr<LoadBalancer> loadBalancer;
    MetricsTracker metricsTracker;

    void solveHeterogeneous();

private:
    // gpu data structures
    conf::fp_type* A_gpu;
    conf::fp_type* b_gpu;
    conf::fp_type* x_gpu;
    conf::fp_type* r_gpu;
    conf::fp_type* d_gpu;
    conf::fp_type* q_gpu;
    conf::fp_type* tmp_gpu;

    // cpu data structures
    conf::fp_type* r_cpu;
    conf::fp_type* d_cpu;
    conf::fp_type* q_cpu;
    conf::fp_type* tmp_cpu;

    // variables
    std::size_t blockCountGPU;
    std::size_t blockCountCPU;
    std::size_t blockStartCPU;

    std::size_t maxBlockCountGPU; /// maximum number of blocks in X/Y direction for the GPU


    void rebalanceProportions(double &gpuProportion);

    void initGPUdataStructures();

    void initCPUdataStructures();

    void freeDataStructures();

    void initCG(conf::fp_type& delta_zero, conf::fp_type& delta_new);

    void compute_q();

    void compute_alpha(conf::fp_type& alpha, conf::fp_type& delta_new);

    void update_x(conf::fp_type alpha);

    void computeRealResidual();

    void update_r(conf::fp_type alpha);

    void compute_delta_new(conf::fp_type& delta_new);

    void compute_d(conf::fp_type& beta);

    void waitAllQueues();
};



#endif //CG_HPP
