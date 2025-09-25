#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_RUNTIMELOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_RUNTIMELOADBALANCER_HPP

#include "LoadBalancer.hpp"

/**
 * This class contains a load balancer implementation that determines the GPU proportion based on kernel runtime metrics.
 */
class RuntimeLoadBalancer : public LoadBalancer{
public:
    RuntimeLoadBalancer(int updateInterval, double initialProportionGPU, int blockCountXY);

    /**
     * Determines the new GPU proportion based on the kernel runtime metrics obtained int the last update interval.
     *
     * @param metricsTracker metrics tracker for the kernel runtime metrics (among others)
     * @return the new GPU proportion
     */
    double getNewProportionGPU(MetricsTracker &metricsTracker) override;

};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_RUNTIMELOADBALANCER_HPP
