#include "PowerLoadBalancer.hpp"

#include <iostream>
#include <numeric>

PowerLoadBalancer::PowerLoadBalancer(const int updateInterval, const double initialProportionGPU, int blockCountXY): LoadBalancer(
    updateInterval, initialProportionGPU, blockCountXY) {
}

double PowerLoadBalancer::getNewProportionGPU(MetricsTracker& metricsTracker) {
    if (metricsTracker.blockCounts_GPU.back() == 0 || metricsTracker.blockCounts_CPU.back() == 0) {
        // if only one component is used do not reevaluate the proportions
        return currentProportionGPU;
    }


    if (!metricsTracker.powerDraw_GPU.empty() && !metricsTracker.powerDraw_CPU.empty()
        && metricsTracker.matrixVectorTimes_GPU.size() >= static_cast<unsigned long>(updateInterval)
        && metricsTracker.matrixVectorTimes_CPU.size() >= static_cast<unsigned long>(updateInterval)) {
        const std::size_t blockCount_GPU = metricsTracker.blockCounts_GPU.back();
        const std::size_t blockCount_CPU = metricsTracker.blockCounts_CPU.back();

        // Compute Power statistics
        double watts_GPU = metricsTracker.powerDraw_GPU.back();
        double watts_CPU = metricsTracker.powerDraw_CPU.back();

        // const double wattsPerBlock_GPU = powerDraw_GPU / static_cast<double>(blockCount_GPU);
        // const double wattsPerBlock_CPU = powerDraw_CPU / static_cast<double>(blockCount_CPU);


        // Compute runtime statistics
        const long offset = static_cast<long>(metricsTracker.matrixVectorTimes_GPU.size()) - updateInterval;

        const double averageRuntime_GPU = std::accumulate(metricsTracker.matrixVectorTimes_GPU.begin() + offset,
                                                          metricsTracker.matrixVectorTimes_GPU.end(),
                                                          0.0) / updateInterval;
        const double averageRuntime_CPU = std::accumulate(metricsTracker.matrixVectorTimes_CPU.begin() + offset,
                                                          metricsTracker.matrixVectorTimes_CPU.end(),
                                                          0.0) / updateInterval;


        const double runtimePerBlock_GPU = averageRuntime_GPU / static_cast<double>(blockCount_GPU);
        const double runtimePerBlock_CPU = averageRuntime_CPU / static_cast<double>(blockCount_CPU);


        // Compare energy efficiency for scenarios gpu-only, cpu-only and heterogeneous

        const double blockCount_total = static_cast<double>(blockCount_GPU) + static_cast<double>(blockCount_CPU);

        const double joulesGPUOnly = watts_GPU * runtimePerBlock_GPU * blockCount_total + conf::idleWatt_CPU *
            runtimePerBlock_GPU * blockCount_total;
        const double joulesCPUOnly = watts_CPU * runtimePerBlock_CPU * blockCount_total;


        const double proportionGPU_heterogeneous = (conf::runtimeLBFactorCPU * runtimePerBlock_CPU) / (
            conf::runtimeLBFactorCPU * runtimePerBlock_CPU + runtimePerBlock_GPU);
        const double blockCountGPU_heterogeneous = std::ceil(blockCount_total * proportionGPU_heterogeneous);
        const double blockCountCPU_heterogeneous = blockCount_total - blockCountGPU_heterogeneous;

        const double joulesHeterogeneous = watts_GPU * runtimePerBlock_GPU * blockCountGPU_heterogeneous +
            watts_CPU * runtimePerBlock_CPU * blockCountCPU_heterogeneous;

        std::cout << "Joules GPU: " << joulesGPUOnly << std::endl;
        std::cout << "Joules CPU: " << joulesCPUOnly << std::endl;
        std::cout << "Joules Heterogeneous: " << joulesHeterogeneous << std::endl;


        if (joulesGPUOnly < joulesHeterogeneous && joulesGPUOnly < joulesCPUOnly) {
            // GPU-only is most energy-efficient
            currentProportionGPU = 1.0;
        } else if (joulesHeterogeneous < joulesGPUOnly && joulesHeterogeneous < joulesCPUOnly) {
            // Heterogeneous is most energy-efficient
            currentProportionGPU = proportionGPU_heterogeneous;
        } else {
            // CPU-only is most energy-efficient
            currentProportionGPU = 0.0;
        }

        return currentProportionGPU;
    } else {
        // return old proportion as fallback
        return currentProportionGPU;
    }
}
