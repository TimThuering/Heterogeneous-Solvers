#include <iostream>
#include "StaticLoadBalancer.hpp"

#include <iostream>

StaticLoadBalancer::StaticLoadBalancer(int updateInterval, double gpuProportion, int blockCountXY) : LoadBalancer(updateInterval, gpuProportion, blockCountXY), gpuProportion(gpuProportion) {
    if (gpuProportion > 1.0 || gpuProportion < 0.0) {
        throw std::runtime_error("Invalid GPU proportion. Must be between 0 and 1");
    }
}

double StaticLoadBalancer::getNewProportionGPU(MetricsTracker &metricsTracker) {
    if (conf::algorithm == "cholesky" && metricsTracker.blockCounts_GPU.back() + metricsTracker.blockCounts_CPU.back() <= conf::blockCountCholeskyGPU_only) {
        gpuProportion = 1;
    }
    return gpuProportion;
}

