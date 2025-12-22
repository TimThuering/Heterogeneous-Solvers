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

/**
 * This class contains the heterogeneous implementation of the CG algorithm that solves Ax = b.
 */
class CG {
public:
    CG(SymmetricMatrix& A,RightHandSide& b, queue& cpuQueue, queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A; /// SPD matrix A
    RightHandSide& b; /// right hand side b

    std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>> x; /// result vector of x of the linear system Ax = b

    queue& cpuQueue; /// SYCL queue for the CPU device
    queue& gpuQueue; /// SYCL queue for the GPU device

    std::shared_ptr<LoadBalancer> loadBalancer; /// load balancer to dynamically or statically determine the CPU/GPU split
    MetricsTracker metricsTracker; /// metrics tracker that tracks various runtime metrics

    void solveHeterogeneous(); /// main method that starts the solver

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

    /**
     * Operation that shifts the horizontal split between the CPU and GPU and ensures that all data structures are consistent.
     *
     * @param gpuProportion
     */
    void rebalanceProportions(double &gpuProportion);

    /**
     * Operation that initializes the GPU data structures
     */
    void initGPUdataStructures();

    /**
     * Operation that initializes the CPU data structures
     */
    void initCPUdataStructures();

    /**
     * Operation that frees all data structures
     */
    void freeDataStructures();

    /**
     * Method that performs the initial setup of the CG algorithm before the iterations start.
     *
     * @param delta_zero reference to delta_zero value
     * @param delta_new reference to delta_new value
     */
    void initCG(conf::fp_type& delta_zero, conf::fp_type& delta_new);

    /**
     * Computes the step q = Ad
     */
    void compute_q();

    /**
     * Computes the step 𝛼 = δ_new / d^T * q
     *
     * @param alpha reference to alpha value
     * @param delta_new reference to delta_new value
     */
    void compute_alpha(conf::fp_type& alpha, conf::fp_type& delta_new);

    /**
     * Computes x = x + 𝛼d
     *
     * @param alpha reference to alpha value
     */
    void update_x(conf::fp_type alpha);

    /**
     * Computes r = b - Ax, i.e., the real residual
     */
    void computeRealResidual();

    /**
     * Computes r = r - 𝛼q, i.e., the residual is updated based on alpha and q
     *
     * @param alpha the current alpha value
     */
    void update_r(conf::fp_type alpha);

    /**
     * Computes δ_new = r^T * r
     *
     * @param delta_new reference to the delta new value
     */
    void compute_delta_new(conf::fp_type& delta_new);

    /**
     * Computes d = r + βd
     *
     * @param beta reference to the beta value
     */
    void compute_d(conf::fp_type& beta);

    /**
     * Operation that waits on the CPU and GPU queue
     */
    void waitAllQueues();
};



#endif //CG_HPP
