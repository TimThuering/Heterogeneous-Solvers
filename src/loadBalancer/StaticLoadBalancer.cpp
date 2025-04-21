#include <stdexcept>
#include "StaticLoadBalancer.hpp"

StaticLoadBalancer::StaticLoadBalancer(conf::fp_type gpuProportion, int updateInterval) : LoadBalancer(updateInterval), gpuProportion(gpuProportion) {
    if (gpuProportion > 1.0 || gpuProportion < 0.0) {
        throw std::runtime_error("Invalid GPU proportion. Must be between 0 and 1");
    }
}

conf::fp_type StaticLoadBalancer::getNewProportionGPU() {
    return gpuProportion;
}

