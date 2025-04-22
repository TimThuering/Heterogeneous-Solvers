#include <iostream>
#include "UtilizationLoadBalancer.hpp"
#include "hws/cpu/cpu_samples.hpp"
#include "hws/cpu/hardware_sampler.hpp"
#include "hws/gpu_nvidia/hardware_sampler.hpp"
#include "hws/gpu_nvidia/nvml_samples.hpp"
#include "hws/gpu_amd/hardware_sampler.hpp"
#include "hws/gpu_intel/hardware_sampler.hpp"

UtilizationLoadBalancer::UtilizationLoadBalancer(int updateInterval, double initialProportionGPU) : LoadBalancer(updateInterval, initialProportionGPU){

}

double UtilizationLoadBalancer::getNewProportionGPU(MetricsTracker &metricsTracker) {
    if (metricsTracker.blockCounts_GPU.back() == 0 || metricsTracker.blockCounts_CPU.back() == 0) {
        // if only one component is used do not reevaluate the proportions
        return currentProportionGPU;
    }
    double GPU_util = 0;
    double CPU_util = 0;
    if (!metricsTracker.averageUtilization_GPU.empty() && !metricsTracker.averageUtilization_CPU.empty()) {
        GPU_util = metricsTracker.averageUtilization_GPU.back();
        CPU_util = metricsTracker.averageUtilization_CPU.back();
    } else {
        // return old proportion as fallback
        return currentProportionGPU;
    }

    double efficiencyGPU = GPU_util / currentProportionGPU;
    double efficiencyCPU = CPU_util / (1 - currentProportionGPU);

    currentProportionGPU = efficiencyCPU / (efficiencyGPU + efficiencyCPU);

    std::cout << efficiencyCPU / (efficiencyGPU + efficiencyCPU) << std::endl;
    return 1;
//    return currentProportionGPU;
}
