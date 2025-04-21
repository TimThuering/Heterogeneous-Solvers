#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP

#include "Configuration.hpp"
#include "MetricsTracker.hpp"

class LoadBalancer {
public:
    LoadBalancer(int updateInterval, double initialProportionGPU);

    virtual conf::fp_type getNewProportionGPU(MetricsTracker &metricsTracker) = 0;

    int updateInterval;

    double currentProportionGPU;

    virtual ~LoadBalancer() {}
};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP
