#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_METRICSTRACKER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_METRICSTRACKER_HPP

#include "Configuration.hpp"
#include "hws/cpu/cpu_samples.hpp"
#include "hws/cpu/hardware_sampler.hpp"
#include "hws/gpu_nvidia/hardware_sampler.hpp"
#include "hws/gpu_nvidia/nvml_samples.hpp"
#include "hws/gpu_amd/hardware_sampler.hpp"
#include "hws/gpu_intel/hardware_sampler.hpp"
#include "hws/system_hardware_sampler.hpp"

class MetricsTracker {
public:
    hws::system_hardware_sampler sampler{conf::sampleCategories};

    std::vector<double> averageUtilization_GPU;
    std::vector<double> averageUtilization_CPU;

    std::vector<double> powerDraw_GPU;
    std::vector<double> powerDraw_CPU;

    std::vector<std::size_t> blockCounts_GPU;
    std::vector<std::size_t> blockCounts_CPU;

    std::vector<double> matrixVectorTimes_GPU;
    std::vector<double> matrixVectorTimes_CPU;

    std::vector<double> times_q;
    std::vector<double> times_alpha;
    std::vector<double> times_x;
    std::vector<double> times_r;
    std::vector<double> times_delta;
    std::vector<double> times_d;

    std::vector<double> memcopy_d;

    std::vector<double> iterationTimes;

    void startTracking();

    void endTracking();

    void
    updateMetrics(std::size_t iteration, std::size_t blockCount_GPU, std::size_t blockCount_CPU, double iterationTime,
                  int updateInterval);

    void writeJSON(std::string &path);

private:
    std::size_t nextTimePoint_GPU = 0;
    std::size_t nextTimePoint_CPU = 0;

    std::size_t nextTimePointPower_GPU = 0;
    std::size_t nextTimePointPower_CPU = 0;

    template<typename T>
    std::string vectorToJSONString(std::vector<T> vector);


};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_METRICSTRACKER_HPP
