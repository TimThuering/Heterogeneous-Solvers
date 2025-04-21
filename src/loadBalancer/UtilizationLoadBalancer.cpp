#include "UtilizationLoadBalancer.hpp"
#include "hws/cpu/cpu_samples.hpp"
#include "hws/cpu/hardware_sampler.hpp"
#include "hws/gpu_nvidia/hardware_sampler.hpp"
#include "hws/gpu_nvidia/nvml_samples.hpp"
#include "hws/gpu_amd/hardware_sampler.hpp"
#include "hws/gpu_intel/hardware_sampler.hpp"

UtilizationLoadBalancer::UtilizationLoadBalancer(int updateInterval) : LoadBalancer(updateInterval){

}

conf::fp_type UtilizationLoadBalancer::getNewProportionGPU(MetricsTracker &metricsTracker) {
    double GPU_util = metricsTracker.averageUtilization_GPU.back();
    double CPU_util = metricsTracker.averageUtilization_CPU.back();

    
    return 1;
}
