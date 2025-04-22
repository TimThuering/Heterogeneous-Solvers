#include <numeric>
#include "RuntimeLoadBalancer.hpp"

RuntimeLoadBalancer::RuntimeLoadBalancer(int updateInterval, double initialProportionGPU) : LoadBalancer(updateInterval,
                                                                                                         initialProportionGPU) {

}

double RuntimeLoadBalancer::getNewProportionGPU(MetricsTracker &metricsTracker) {

    std::size_t offset = metricsTracker.matrixVectorTimes_GPU.size() - updateInterval;

    double averageRuntime_GPU = std::accumulate(metricsTracker.matrixVectorTimes_GPU.begin() + offset, metricsTracker.matrixVectorTimes_GPU.end(),0.0);
    return 0.8;
}
