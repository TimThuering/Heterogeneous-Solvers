#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILIZATIONLOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILIZATIONLOADBALANCER_HPP

#include <hws/system_hardware_sampler.hpp>

#include "LoadBalancer.hpp"
#include "MetricsTracker.hpp"

class UtilizationLoadBalancer : public LoadBalancer {

public:
    UtilizationLoadBalancer(int updateInterval);


    conf::fp_type getNewProportionGPU(MetricsTracker &metricsTracker) override;

};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILIZATIONLOADBALANCER_HPP
