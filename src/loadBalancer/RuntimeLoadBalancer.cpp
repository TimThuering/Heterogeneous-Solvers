#include <numeric>
#include <iostream>
#include "RuntimeLoadBalancer.hpp"

RuntimeLoadBalancer::RuntimeLoadBalancer(int updateInterval, double initialProportionGPU) : LoadBalancer(updateInterval,
                                                                                                         initialProportionGPU) {

}

double RuntimeLoadBalancer::getNewProportionGPU(MetricsTracker &metricsTracker) {
    if (metricsTracker.blockCounts_GPU.back() == 0 || metricsTracker.blockCounts_CPU.back() == 0) {
        // if only one component is used do not reevaluate the proportions
        return currentProportionGPU;
    }
    if (metricsTracker.matrixVectorTimes_GPU.size() >= static_cast<unsigned long>(updateInterval) &&
        metricsTracker.matrixVectorTimes_CPU.size() >= static_cast<unsigned long>(updateInterval)) {


        const long offset = static_cast<long>(metricsTracker.matrixVectorTimes_GPU.size()) - updateInterval;

        const double averageRuntime_GPU = std::accumulate(metricsTracker.matrixVectorTimes_GPU.begin() + offset,
                                                    metricsTracker.matrixVectorTimes_GPU.end(), 0.0) / updateInterval;
        const double averageRuntime_CPU = std::accumulate(metricsTracker.matrixVectorTimes_CPU.begin() + offset,
                                                    metricsTracker.matrixVectorTimes_CPU.end(), 0.0) / updateInterval;

        const std::size_t blockCount_GPU = metricsTracker.blockCounts_GPU.back();
        const std::size_t blockCount_CPU = metricsTracker.blockCounts_CPU.back();

        const double runtimePerBlock_GPU = averageRuntime_GPU / static_cast<double>(blockCount_GPU);
        const double runtimePerBlock_CPU = conf::runtimeLBFactorCPU * (averageRuntime_CPU / static_cast<double>(blockCount_CPU));

        // std::cout << "GPU runtime " << averageRuntime_GPU << std::endl;
        // std::cout << "CPU runtime " << averageRuntime_CPU << std::endl;

        const double newProportionGPU = runtimePerBlock_CPU / (runtimePerBlock_CPU + runtimePerBlock_GPU);

        currentProportionGPU = newProportionGPU;
        return currentProportionGPU;
    } else {
        return currentProportionGPU;
    }
}
