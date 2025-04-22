
#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP

#include "LoadBalancer.hpp"

class StaticLoadBalancer : public LoadBalancer {

public:
    StaticLoadBalancer(int updateInterval, double gpuProportion);

    double gpuProportion;

    double getNewProportionGPU(MetricsTracker &metricsTracker) override;

    ~StaticLoadBalancer() {}

};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP
