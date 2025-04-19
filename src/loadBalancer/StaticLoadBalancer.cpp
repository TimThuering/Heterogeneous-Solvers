#include "StaticLoadBalancer.hpp"

StaticLoadBalancer::StaticLoadBalancer(conf::fp_type gpuProportion, int updateInterval) : LoadBalancer(updateInterval), gpuProportion(gpuProportion) {
    if (gpuProportion > 1.0 || gpuProportion < 0.0) {
        
    }
}

conf::fp_type StaticLoadBalancer::getNewProportionGPU() {
    return gpuProportion;
}

