#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_METRICSTRACKER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_METRICSTRACKER_HPP

#include "hws/cpu/cpu_samples.hpp"
#include "hws/cpu/hardware_sampler.hpp"
#include "hws/gpu_nvidia/hardware_sampler.hpp"
#include "hws/gpu_nvidia/nvml_samples.hpp"
#include "hws/gpu_amd/hardware_sampler.hpp"
#include "hws/gpu_intel/hardware_sampler.hpp"
#include "hws/system_hardware_sampler.hpp"

class MetricsTracker {
public:
    hws::system_hardware_sampler utilizationSampler{hws::sample_category::general};
    hws::system_hardware_sampler powerSampler{hws::sample_category::power};

    std::vector<double> averageUtilization_GPU;
    std::vector<double> averageUtilization_CPU;

    std::vector<double> averagePowerDraw_GPU;
    std::vector<double> averagePowerDraw_CPU;

    std::vector<std::size_t > blockCounts_GPU;
    std::vector<std::size_t > blockCounts_CPU;

    std::vector<double> matrixVectorTimes_GPU;
    std::vector<double> matrixVectorTimes_CPU;

    std::vector<double> iterationTimes;

    void startTracking();

    void endTracking();

    void updateMetrics(std::size_t iteration, std::size_t  blockCount_GPU, std::size_t  blockCount_CPU, double iterationTime, int updateInterval);
private:
    std::size_t nextTimePoint_GPU = 0;
    std::size_t nextTimePoint_CPU = 0;


};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_METRICSTRACKER_HPP
