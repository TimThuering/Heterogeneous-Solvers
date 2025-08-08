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

    bool condition = false;
    if (conf::algorithm == "cg") {
        condition = !metricsTracker.powerDraw_GPU.empty() && !metricsTracker.powerDraw_CPU.empty() && metricsTracker.matrixVectorTimes_GPU.size() >= static_cast<unsigned long>(updateInterval) && metricsTracker.matrixVectorTimes_CPU.size() >= static_cast<unsigned long>(updateInterval);
    } else if (conf::algorithm == "cholesky") {
        condition = !metricsTracker.powerDraw_GPU.empty() && !metricsTracker.powerDraw_CPU.empty() && metricsTracker.matrixMatrixTimes_GPU.size() >= static_cast<unsigned long>(updateInterval) && metricsTracker.matrixMatrixTimes_CPU.size() >= static_cast<unsigned long>(updateInterval);
    }

    if (condition) {
        const std::size_t blockCount_GPU = metricsTracker.blockCounts_GPU.back();
        const std::size_t blockCount_CPU = metricsTracker.blockCounts_CPU.back();

        // Compute Power statistics
        double watts_GPU = metricsTracker.powerDraw_GPU.back();
        double watts_CPU = metricsTracker.powerDraw_CPU.back();

        double runtimePerBlock_GPU = 0.0;
        double runtimePerBlock_CPU = 0.0;
        double communicationTimePerBlock = 0.0;
        double shiftTimePerBlock = 0.0;


        // Compute runtime statistics
        if (conf::algorithm == "cg") {
            const long offset = static_cast<long>(metricsTracker.matrixVectorTimes_GPU.size()) - updateInterval;

            const double averageRuntime_GPU = std::accumulate(metricsTracker.matrixVectorTimes_GPU.begin() + offset, metricsTracker.matrixVectorTimes_GPU.end(), 0.0) / updateInterval;
            const double averageRuntime_CPU = std::accumulate(metricsTracker.matrixVectorTimes_CPU.begin() + offset, metricsTracker.matrixVectorTimes_CPU.end(), 0.0) / updateInterval;

            runtimePerBlock_GPU = averageRuntime_GPU / static_cast<double>(blockCount_GPU);
            runtimePerBlock_CPU = averageRuntime_CPU / static_cast<double>(blockCount_CPU);
        } else if (conf::algorithm == "cholesky") {
            const long offset = static_cast<long>(metricsTracker.matrixMatrixTimes_GPU.size()) - updateInterval;

            std::vector<double> timesGPU(metricsTracker.matrixMatrixTimes_GPU.begin() + offset, metricsTracker.matrixMatrixTimes_GPU.end());
            std::vector<double> timesCPU(metricsTracker.matrixMatrixTimes_CPU.begin() + offset, metricsTracker.matrixMatrixTimes_CPU.end());
            std::vector<double> timesShift(metricsTracker.shiftTimes.begin() + offset, metricsTracker.shiftTimes.end());
            std::vector<double> timesCommunication(metricsTracker.copyTimes.begin() + offset, metricsTracker.copyTimes.end());

            std::vector<std::size_t> blocksGPU(metricsTracker.blockCounts_GPU.begin() + offset, metricsTracker.blockCounts_GPU.end());
            std::vector<std::size_t> blocksCPU(metricsTracker.blockCounts_CPU.begin() + offset, metricsTracker.blockCounts_CPU.end());

            // convert absolute time to time per block
            for (std::size_t i = 0; i < timesGPU.size(); i++) {
                const double verticalBlockCount_CPU = static_cast<double>(blocksCPU[i]);
                const double totalBlockCount_CPU = ((verticalBlockCount_CPU - 1) * verticalBlockCount_CPU) / 2.0;

                const double verticalBlockCount_GPU = static_cast<double>(blocksGPU[i]);
                const double totalBlockCount = ((verticalBlockCount_GPU + verticalBlockCount_CPU - 1) * (verticalBlockCount_GPU + verticalBlockCount_CPU)) / 2.0;
                const double totalBlockCount_GPU = totalBlockCount - totalBlockCount_CPU;

                timesCPU[i] = timesCPU[i] / totalBlockCount_CPU;
                timesGPU[i] = timesGPU[i] / totalBlockCount_GPU;
                timesShift[i] = timesShift[i] / (verticalBlockCount_CPU + verticalBlockCount_GPU);
                timesCommunication[i] = timesCommunication[i] / (verticalBlockCount_CPU);

            }

            runtimePerBlock_GPU = std::accumulate(timesGPU.begin(), timesGPU.end(), 0.0) / updateInterval;
            runtimePerBlock_CPU = std::accumulate(timesCPU.begin(), timesCPU.end(), 0.0) / updateInterval;
            shiftTimePerBlock = std::accumulate(timesShift.begin(), timesShift.end(), 0.0) / updateInterval;
            communicationTimePerBlock = std::accumulate(timesCommunication.begin(), timesCommunication.end(), 0.0) / updateInterval;
        }

        // Compare energy efficiency for scenarios gpu-only, cpu-only and heterogeneous
        double blockCount_total = 0.0;
        double proportionGPU_heterogeneous = 1.0;
        double totalVerticalBlockCountNextIteration = 0.0;

        double newVerticalBlockCount_CPU = 0.0;

        if (conf::algorithm == "cg") {
            blockCount_total = static_cast<double>(blockCount_GPU) + static_cast<double>(blockCount_CPU);
            proportionGPU_heterogeneous = (conf::runtimeLBFactorCPU * runtimePerBlock_CPU) / (conf::runtimeLBFactorCPU * runtimePerBlock_CPU + runtimePerBlock_GPU);
        } else if (conf::algorithm == "cholesky") {
            totalVerticalBlockCountNextIteration = static_cast<double>(blockCountXY - metricsTracker.matrixMatrixTimes_GPU.size() - 2);
            blockCount_total = (totalVerticalBlockCountNextIteration * (totalVerticalBlockCountNextIteration + 1)) / 2.0;

            proportionGPU_heterogeneous = (conf::runtimeLBFactorCPU * runtimePerBlock_CPU) / (conf::runtimeLBFactorCPU * runtimePerBlock_CPU + runtimePerBlock_GPU);
            const double newTotalBlockCount_CPU = std::floor(blockCount_total * (1 - proportionGPU_heterogeneous));

            newVerticalBlockCount_CPU = std::floor((-1 + std::sqrt(1 + 8 * newTotalBlockCount_CPU)) / 2.0);
            double newVerticalProportionCPU = newVerticalBlockCount_CPU / totalVerticalBlockCountNextIteration;
            proportionGPU_heterogeneous = 1 - newVerticalProportionCPU;
        }

        const double joulesShift = (watts_GPU + watts_CPU) * shiftTimePerBlock * totalVerticalBlockCountNextIteration;
        const double joulesCommunication = (watts_GPU + watts_CPU) * communicationTimePerBlock * newVerticalBlockCount_CPU;

        const double joulesGPUOnly = watts_GPU * runtimePerBlock_GPU * blockCount_total + conf::idleWatt_CPU * runtimePerBlock_GPU * blockCount_total;
        const double joulesCPUOnly = watts_CPU * runtimePerBlock_CPU * conf::energyLBFactorCPU * blockCount_total;

        const double blockCountCPU_heterogeneous = (newVerticalBlockCount_CPU * (newVerticalBlockCount_CPU + 1) )/ 2.0;
        const double blockCountGPU_heterogeneous = blockCount_total - blockCountCPU_heterogeneous;

        const double joulesHeterogeneous = watts_GPU * runtimePerBlock_GPU * blockCountGPU_heterogeneous + watts_CPU * runtimePerBlock_CPU * conf::runtimeLBFactorCPU * blockCountCPU_heterogeneous + joulesCommunication + joulesShift;

        if (conf::printVerbose) {
            std::cout << "Joules GPU: " << joulesGPUOnly << std::endl;
            std::cout << "Joules CPU: " << joulesCPUOnly << std::endl;
            std::cout << "Joules Heterogeneous: " << joulesHeterogeneous << std::endl;
            std::cout << "Joules comm: " << joulesCommunication << std::endl;
            std::cout << "Joules shift: " << joulesShift << std::endl;
            std::cout << "heterogeneous gpu proportion: " << proportionGPU_heterogeneous << std::endl;

            std::cout << "expected heterogeneous Time: " <<  std::max(runtimePerBlock_GPU * blockCountGPU_heterogeneous , runtimePerBlock_CPU * conf::runtimeLBFactorCPU *  blockCountCPU_heterogeneous )<< std::endl;
            std::cout << "expected CPU Time: " <<  runtimePerBlock_CPU * blockCount_total << std::endl;
            std::cout << "expected GPU Time: " <<  runtimePerBlock_GPU * blockCount_total << std::endl;
        }


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
