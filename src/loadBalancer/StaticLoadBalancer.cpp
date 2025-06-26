#include <stdexcept>
#include "StaticLoadBalancer.hpp"

StaticLoadBalancer::StaticLoadBalancer(int updateInterval, double gpuProportion, int blockCountXY) : LoadBalancer(updateInterval, gpuProportion, blockCountXY), gpuProportion(gpuProportion) {
    if (gpuProportion > 1.0 || gpuProportion < 0.0) {
        throw std::runtime_error("Invalid GPU proportion. Must be between 0 and 1");
    }
}

double StaticLoadBalancer::getNewProportionGPU(MetricsTracker &metricsTracker) {
    return gpuProportion;
}

