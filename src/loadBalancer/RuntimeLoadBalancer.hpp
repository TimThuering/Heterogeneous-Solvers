#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_RUNTIMELOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_RUNTIMELOADBALANCER_HPP

#include "LoadBalancer.hpp"

class RuntimeLoadBalancer : public LoadBalancer{
public:
    RuntimeLoadBalancer(int updateInterval, double initialProportionGPU, int blockCountXY);

    double getNewProportionGPU(MetricsTracker &metricsTracker) override;

};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_RUNTIMELOADBALANCER_HPP
