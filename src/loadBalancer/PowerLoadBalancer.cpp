#include "PowerLoadBalancer.hpp"

PowerLoadBalancer::PowerLoadBalancer(const int updateInterval, const double initialProportionGPU): LoadBalancer(updateInterval, initialProportionGPU) {
}

double PowerLoadBalancer::getNewProportionGPU(MetricsTracker& metricsTracker) {
    if (metricsTracker.blockCounts_GPU.back() == 0 || metricsTracker.blockCounts_CPU.back() == 0) {
        // if only one component is used do not reevaluate the proportions
        return currentProportionGPU;
    }

    double powerDraw_GPU = 0;
    double powerDraw_CPU = 0;
    if (!metricsTracker.powerDraw_GPU.empty() && !metricsTracker.powerDraw_CPU.empty()) {
        powerDraw_GPU = metricsTracker.powerDraw_GPU.back();
        powerDraw_CPU = metricsTracker.powerDraw_CPU.back();

        const std::size_t blockCount_GPU = metricsTracker.blockCounts_GPU.back();
        const std::size_t blockCount_CPU = metricsTracker.blockCounts_CPU.back();

        const double wattsPerBlock_GPU = 1.0 / (powerDraw_GPU / static_cast<double>(blockCount_GPU));
        const double wattsPerBlock_CPU = 1.0 / (powerDraw_CPU / static_cast<double>(blockCount_CPU));

        const double newProportionGPU = wattsPerBlock_GPU / (wattsPerBlock_CPU + wattsPerBlock_GPU);

        currentProportionGPU = newProportionGPU;
        return currentProportionGPU;
    } else {
        // return old proportion as fallback
        return currentProportionGPU;
    }
}
