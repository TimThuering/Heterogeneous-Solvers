
#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP

#include "LoadBalancer.hpp"

/**
 * This class contains a load balancer implementation that is used when a static CPU/GPU split is desired.
 */
class StaticLoadBalancer : public LoadBalancer {

public:
    StaticLoadBalancer(int updateInterval, double gpuProportion, int blockCountXY);

    double gpuProportion;

    /**
     * Always returns the same static GPU proportion as configured at setup.
     *
     * @param metricsTracker metrics tracker
     * @return the new GPU proportion
     */
    double getNewProportionGPU(MetricsTracker &metricsTracker) override;

    ~StaticLoadBalancer() {}

};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP
