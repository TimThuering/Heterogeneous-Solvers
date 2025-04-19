#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP

#include "Configuration.hpp"

class LoadBalancer {
public:
    LoadBalancer(int updateInterval);

    virtual conf::fp_type getNewProportionGPU() = 0;

    int updateInterval;

    virtual ~LoadBalancer() {}
};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP
