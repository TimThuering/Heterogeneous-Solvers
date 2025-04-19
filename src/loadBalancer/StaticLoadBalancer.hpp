
#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP

#include "LoadBalancer.hpp"

class StaticLoadBalancer : public LoadBalancer {

public:
    StaticLoadBalancer(conf::fp_type gpuProportion, int updateInterval);
    conf::fp_type gpuProportion;

    conf::fp_type getNewProportionGPU() override;

    ~StaticLoadBalancer() {}

};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_STATICLOADBALANCER_HPP
