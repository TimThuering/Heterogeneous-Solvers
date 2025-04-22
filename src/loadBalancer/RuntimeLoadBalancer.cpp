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
    if (metricsTracker.matrixVectorTimes_GPU.size() >= updateInterval &&
        metricsTracker.matrixVectorTimes_CPU.size() >= updateInterval) {


        std::size_t offset = metricsTracker.matrixVectorTimes_GPU.size() - updateInterval;

        double averageRuntime_GPU = std::accumulate(metricsTracker.matrixVectorTimes_GPU.begin() + offset,
                                                    metricsTracker.matrixVectorTimes_GPU.end(), 0.0) / updateInterval;
        double averageRuntime_CPU = std::accumulate(metricsTracker.matrixVectorTimes_CPU.begin() + offset,
                                                    metricsTracker.matrixVectorTimes_CPU.end(), 0.0) / updateInterval;

        std::size_t blockCount_GPU = metricsTracker.blockCounts_GPU.back();
        std::size_t blockCount_CPU = metricsTracker.blockCounts_CPU.back();

        double runtimePerBlock_GPU = averageRuntime_GPU / static_cast<double>(blockCount_GPU);
        double runtimePerBlock_CPU = averageRuntime_CPU / static_cast<double>(blockCount_CPU);

        std::cout << "GPU runtime " << averageRuntime_GPU << std::endl;
        std::cout << "CPU runtime " << averageRuntime_CPU << std::endl;


        return (runtimePerBlock_CPU / runtimePerBlock_GPU) / ((runtimePerBlock_CPU / runtimePerBlock_GPU) + 1);
    } else {
        return currentProportionGPU;
    }
}
