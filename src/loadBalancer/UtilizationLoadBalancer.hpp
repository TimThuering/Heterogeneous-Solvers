#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILIZATIONLOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILIZATIONLOADBALANCER_HPP

#include <hws/system_hardware_sampler.hpp>

#include "LoadBalancer.hpp"
class UtilizationLoadBalancer : public LoadBalancer {

public:
    UtilizationLoadBalancer(int updateInterval);

    hws::system_hardware_sampler cpuUtilSampler;

};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILIZATIONLOADBALANCER_HPP
