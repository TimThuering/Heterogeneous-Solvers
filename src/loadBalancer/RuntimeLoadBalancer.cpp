#include <numeric>
#include <iostream>
#include "RuntimeLoadBalancer.hpp"

RuntimeLoadBalancer::RuntimeLoadBalancer(int updateInterval, double initialProportionGPU, int blockCountXY) : LoadBalancer(updateInterval,
                                                                                                         initialProportionGPU, blockCountXY) {
}

double RuntimeLoadBalancer::getNewProportionGPU(MetricsTracker& metricsTracker) {
    if (metricsTracker.blockCounts_GPU.back() == 0 || metricsTracker.blockCounts_CPU.back() == 0) {
        // if only one component is used do not reevaluate the proportions
        return currentProportionGPU;
    }
    if (conf::algorithm == "cg") {
        if (metricsTracker.matrixVectorTimes_GPU.size() >= static_cast<unsigned long>(updateInterval) && metricsTracker.matrixVectorTimes_CPU.size() >= static_cast<unsigned long>(updateInterval)) {
            const long offset = static_cast<long>(metricsTracker.matrixVectorTimes_GPU.size()) - updateInterval;

            const double averageRuntime_GPU = std::accumulate(metricsTracker.matrixVectorTimes_GPU.begin() + offset, metricsTracker.matrixVectorTimes_GPU.end(), 0.0) / updateInterval;
            const double averageRuntime_CPU = std::accumulate(metricsTracker.matrixVectorTimes_CPU.begin() + offset, metricsTracker.matrixVectorTimes_CPU.end(), 0.0) / updateInterval;

            const std::size_t blockCount_GPU = metricsTracker.blockCounts_GPU.back();
            const std::size_t blockCount_CPU = metricsTracker.blockCounts_CPU.back();

            const double runtimePerBlock_GPU = averageRuntime_GPU / static_cast<double>(blockCount_GPU);
            const double runtimePerBlock_CPU = conf::runtimeLBFactorCPU * (averageRuntime_CPU / static_cast<double>(blockCount_CPU));


            const double newProportionGPU = runtimePerBlock_CPU / (runtimePerBlock_CPU + runtimePerBlock_GPU);

            currentProportionGPU = newProportionGPU;
            return currentProportionGPU;
        }
        return currentProportionGPU;
    } else if (conf::algorithm == "cholesky") {
        if (metricsTracker.matrixMatrixTimes_GPU.size() >= static_cast<unsigned long>(updateInterval) && metricsTracker.matrixMatrixTimes_CPU.size() >= static_cast<unsigned long>(updateInterval)) {
            const long offset = static_cast<long>(metricsTracker.matrixMatrixTimes_GPU.size()) - updateInterval;

            std::vector<double> timesGPU(metricsTracker.matrixMatrixTimes_GPU.begin() + offset, metricsTracker.matrixMatrixTimes_GPU.end());
            std::vector<double> timesCPU(metricsTracker.matrixMatrixTimes_CPU.begin() + offset, metricsTracker.matrixMatrixTimes_CPU.end());

            std::vector<std::size_t> blocksGPU(metricsTracker.blockCounts_GPU.begin() + offset, metricsTracker.blockCounts_GPU.end());
            std::vector<std::size_t> blocksCPU(metricsTracker.blockCounts_CPU.begin() + offset, metricsTracker.blockCounts_CPU.end());

            // convert absolute time to time per block
            for (std::size_t i = 0; i < timesGPU.size(); i++) {
                const double verticalBlockCount_CPU = static_cast<double>(metricsTracker.blockCounts_CPU[i]);
                const double totalBlockCount_CPU = ((verticalBlockCount_CPU - 1) * verticalBlockCount_CPU) / 2.0;

                const double verticalBlockCount_GPU = static_cast<double>(metricsTracker.blockCounts_GPU[i]);
                const double totalBlockCount = ((verticalBlockCount_GPU + verticalBlockCount_CPU - 1) * (verticalBlockCount_GPU + verticalBlockCount_CPU)) / 2.0;
                const double totalBlockCount_GPU = totalBlockCount - totalBlockCount_CPU;

                timesCPU[i] = timesCPU[i] / totalBlockCount_CPU;
                timesGPU[i] = timesGPU[i] / totalBlockCount_GPU;
            }


            const double totalVerticalBlockCountNextIteration = static_cast<double>(blockCountXY - metricsTracker.matrixMatrixTimes_GPU.size() - 2);
            const double totalBlockCountNextIteration = (totalVerticalBlockCountNextIteration * (totalVerticalBlockCountNextIteration + 1)) / 2;

            const double averageRuntimePerBlock_GPU = std::accumulate(timesCPU.begin(), timesCPU.end(), 0.0) / updateInterval;
            const double averageRuntimePerBlock_CPU = conf::runtimeLBFactorCPU * std::accumulate(timesGPU.begin(), timesGPU.end(), 0.0) / updateInterval;

            const double newTotalProportionGPU = averageRuntimePerBlock_CPU / (averageRuntimePerBlock_CPU + averageRuntimePerBlock_GPU);
            const double newTotalBlockCount_CPU = std::floor(totalBlockCountNextIteration * newTotalProportionGPU);
            const double newVerticalBlockCount_CPU = std::floor((-1 + std::sqrt(1 + 8 * newTotalBlockCount_CPU)) / 2);
            const double newVerticalProportionCPU = newVerticalBlockCount_CPU / totalVerticalBlockCountNextIteration;
            const double newVerticalProportionGPU = 1 - newVerticalProportionCPU;

            if (newTotalBlockCount_CPU == 0) {
                return 1;
            }


            std:: cout << "Changing GPU proportion from" << currentProportionGPU << " to " << newVerticalProportionGPU << std::endl;

            currentProportionGPU = newVerticalProportionGPU;
            return currentProportionGPU;
        }
    }
    return currentProportionGPU;
}
