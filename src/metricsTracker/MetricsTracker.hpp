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

    // times of cg algorithm
    std::vector<double> matrixVectorTimes_GPU;
    std::vector<double> matrixVectorTimes_CPU;
    std::vector<double> times_q;
    std::vector<double> times_alpha;
    std::vector<double> times_x;
    std::vector<double> times_r;
    std::vector<double> times_delta;
    std::vector<double> times_d;
    std::vector<double> memcopy_d;

    // times of cholesky algorithm
    std::vector<double> shiftTimes;

    std::vector<double> choleskyDiagonalBlockTimes;

    std::vector<double> copyTimes;

    std::vector<double> triangularSolveTimes_GPU;
    std::vector<double> triangularSolveTimes_CPU;
    std::vector<double> triangularSolveTimes_total;

    std::vector<double> matrixMatrixDiagonalTimes_GPU;
    std::vector<double> matrixMatrixDiagonalTimes_CPU;
    std::vector<double> matrixMatrixDiagonalTimes_total;

    std::vector<double> matrixMatrixTimes_GPU;
    std::vector<double> matrixMatrixTimes_CPU;
    std::vector<double> matrixMatrixTimes_total;

    // total iteration times (time to process one block column in case of cholesky)
    std::vector<double> iterationTimes;

    double memoryInitTime = 0.0;
    double resultCopyTime = 0.0;
    double totalTime = 0.0;

    double solveTime = 0.0; // time for the solver step after the Cholesky decomposition

    void startTracking();

    void endTracking();

    void updateMetrics(std::size_t iteration, std::size_t blockCount_GPU, std::size_t blockCount_CPU, double iterationTime, int updateInterval);

    void writeJSON(std::string& path);

private:
    std::size_t nextTimePoint_GPU = 0;
    std::size_t nextTimePoint_CPU = 0;

    std::size_t nextTimePointPower_GPU = 0;
    std::size_t nextTimePointPower_CPU = 0;

    template <typename T>
    std::string vectorToJSONString(std::vector<T> vector);
};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_METRICSTRACKER_HPP
