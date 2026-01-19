#include <numeric>
#include <iostream>
#include <filesystem>

#include "MetricsTracker.hpp"
#include "UtilityFunctions.hpp"
#include "Configuration.hpp"

void MetricsTracker::updateMetrics(std::size_t iteration, std::size_t blockCount_GPU, std::size_t blockCount_CPU,
                                   double iterationTime,
                                   int updateInterval) {
    sampler.pause_sampling();

    // add new block count of iteration
    blockCounts_GPU.push_back(blockCount_GPU);
    blockCounts_CPU.push_back(blockCount_CPU);

    // add time of iteration
    iterationTimes.push_back(iterationTime);

    // track metrics for load balancing before every update interval
    if ((iteration + 1) % updateInterval == 0) {
        // get samples for power and utilization from the hws library
        auto* cpu_sampler = dynamic_cast<hws::cpu_hardware_sampler*>(sampler.samplers()[0].get());

        hws::cpu_general_samples generalSamples_CPU = cpu_sampler->general_samples();
        hws::cpu_power_samples powerSamples_CPU = cpu_sampler->power_samples();

#ifdef NVIDIA
        auto* gpu_sampler = dynamic_cast<hws::gpu_nvidia_hardware_sampler*>(sampler.samplers()[1].get());
        hws::nvml_general_samples generalSamples_GPU = gpu_sampler->general_samples();
        hws::nvml_power_samples powerSamples_GPU = gpu_sampler->power_samples();
#elif defined(AMD)
        auto* gpu_sampler = dynamic_cast<hws::gpu_amd_hardware_sampler*>(sampler.samplers()[1].get());
        hws::rocm_smi_general_samples generalSamples_GPU = gpu_sampler->general_samples();
        hws::rocm_smi_power_samples powerSamples_GPU = gpu_sampler->power_samples();
#elif defined(INTEL)
        auto* gpu_sampler = dynamic_cast<hws::gpu_intel_hardware_sampler*>(sampler.samplers()[1].get());
        hws::level_zero_general_samples generalSamples_GPU = gpu_sampler->general_samples();
        hws::level_zero_power_samples powerSamples_GPU = gpu_sampler->power_samples();
#endif

#ifndef INTEL
        if (generalSamples_GPU.get_compute_utilization().has_value()) {
            double averageUtil = 0.0;
            if (nextTimePoint_GPU < generalSamples_GPU.get_compute_utilization().value().size()) {
                // at least one new sample is available
                std::vector<double> GPU_util(
                    generalSamples_GPU.get_compute_utilization().value().begin() +
                    static_cast<long>(nextTimePoint_GPU),
                    generalSamples_GPU.get_compute_utilization().value().end());
                averageUtil =
                    std::accumulate(GPU_util.begin(), GPU_util.end(), 0.0) / static_cast<double>(GPU_util.size());
            } else {
                // no new sample was generated in last interval, use last available utilization as fallback
                if (conf::printVerbose) {
                    std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!" << std::endl;
                }
                averageUtil = generalSamples_GPU.get_compute_utilization().value().back();
            }
            // update index where the new samples will begin for the next interval
            nextTimePoint_GPU = generalSamples_GPU.get_compute_utilization().value().size();
            averageUtilization_GPU.push_back(averageUtil);
        }
#endif

        if (generalSamples_CPU.get_compute_utilization().has_value()) {
            double averageUtil = 0.0;
            if (nextTimePoint_CPU < generalSamples_CPU.get_compute_utilization().value().size()) {
                // at least one new sample is available
                std::vector<double> CPU_util(
                    generalSamples_CPU.get_compute_utilization().value().begin() +
                    static_cast<long>(nextTimePoint_CPU),
                    generalSamples_CPU.get_compute_utilization().value().end());
                averageUtil =
                    std::accumulate(CPU_util.begin(), CPU_util.end(), 0.0) / static_cast<double>(CPU_util.size());
            } else {
                // no new sample was generated in last interval, use last available utilization as fallback
                if (conf::printVerbose) {
                    std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!" << std::endl;
                }
                averageUtil = generalSamples_CPU.get_compute_utilization().value().back();
            }
            nextTimePoint_CPU = generalSamples_CPU.get_compute_utilization().value().size();
            averageUtilization_CPU.push_back(averageUtil);
        }

        if (powerSamples_CPU.get_power_usage().has_value()) {
            double powerDraw = 0.0;
            if (nextTimePointPower_CPU < powerSamples_CPU.get_power_usage().value().size()) {
                // at least 1 new power samples is available
                std::vector<double> CPU_watts(
                    powerSamples_CPU.get_power_usage().value().begin() + static_cast<long>(nextTimePointPower_CPU),
                    powerSamples_CPU.get_power_usage().value().end());

                powerDraw = std::accumulate(CPU_watts.begin(), CPU_watts.end(), 0.0) /
                    static_cast<double>(CPU_watts.size());
            } else {
                if (conf::printVerbose) {
                    std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!\n";
                }
                powerDraw = powerDraw_CPU.back();
            }
            nextTimePointPower_CPU = powerSamples_CPU.get_power_usage().value().size();
            powerDraw_CPU.push_back(powerDraw);
        }

        if (powerSamples_GPU.get_power_usage().has_value()) {
            double powerDraw = 0.0;
            if (nextTimePointPower_GPU < powerSamples_GPU.get_power_usage().value().size()) {
                // at least 1 new power samples is available
                std::vector<double> GPU_watts(
                    powerSamples_GPU.get_power_usage().value().begin() + static_cast<long>(nextTimePointPower_GPU),
                    powerSamples_GPU.get_power_usage().value().end());

                powerDraw =
                    std::accumulate(GPU_watts.begin(), GPU_watts.end(), 0.0) / static_cast<double>(GPU_watts.size());
            } else {
                if (conf::printVerbose) {
                    std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!\n";
                }
                powerDraw = powerDraw_GPU.back();
            }
            nextTimePointPower_GPU = powerSamples_GPU.get_power_usage().value().size();
            powerDraw_GPU.push_back(powerDraw);
        }
    }

    sampler.resume_sampling();
}

void MetricsTracker::startTracking() {
    if (conf::enableHWS) {
        sampler.start_sampling();
    }
}

void MetricsTracker::endTracking() {
    if (conf::enableHWS) {
        sampler.stop_sampling();
    }
}

void MetricsTracker::writeJSON(std::string& path) {
    std::ofstream metricsJSON(path + "/metrics.json");
    // get samples for power and utilization from the hws library
    auto* cpu_sampler = dynamic_cast<hws::cpu_hardware_sampler*>(sampler.samplers()[0].get());

    hws::cpu_general_samples generalSamples_CPU = cpu_sampler->general_samples();
    hws::cpu_power_samples powerSamples_CPU = cpu_sampler->power_samples();

#ifdef NVIDIA
    auto* gpu_sampler = dynamic_cast<hws::gpu_nvidia_hardware_sampler*>(sampler.samplers()[1].get());
    hws::nvml_general_samples generalSamples_GPU = gpu_sampler->general_samples();
    hws::nvml_power_samples powerSamples_GPU = gpu_sampler->power_samples();
#elif defined(AMD)
    auto* gpu_sampler = dynamic_cast<hws::gpu_amd_hardware_sampler*>(sampler.samplers()[1].get());
    hws::rocm_smi_general_samples generalSamples_GPU = gpu_sampler->general_samples();
    hws::rocm_smi_power_samples powerSamples_GPU = gpu_sampler->power_samples();
#elif defined(INTEL)
    auto* gpu_sampler = dynamic_cast<hws::gpu_intel_hardware_sampler*>(sampler.samplers()[1].get());
    hws::level_zero_general_samples generalSamples_GPU = gpu_sampler->general_samples();
    hws::level_zero_power_samples powerSamples_GPU = gpu_sampler->power_samples();
#endif

    metricsJSON << "{\n";

    metricsJSON << "\"configuration\": {\n";
    metricsJSON << std::string("\t \"algorithm\":                       ") + "\"" + conf::algorithm + "\"" + ",\n";
    if (cpu_sampler->general_samples().get_name().has_value()) {
        metricsJSON << std::string("\t \"CPU\":                             ") + "\"" + cpu_sampler->general_samples().get_name().value() + "\"" + ",\n";
    }
    if (gpu_sampler->general_samples().get_name().has_value()) {
        metricsJSON << std::string("\t \"GPU\":                             ") + "\"" + gpu_sampler->general_samples().get_name().value() + "\"" + ",\n";
    }

    metricsJSON << "\t \"N\":                               " + std::to_string(conf::N) + ",\n";

#ifdef USE_DOUBLE
    metricsJSON << "\t \"FP_type\":                         " + std::string("\"FP_64\"") + ",\n";
#else
    metricsJSON << "\t \"FP_type\":                         " + std::string("\"FP_32\"") + ",\n";
#endif

    metricsJSON << "\t \"matrixBlockSize\":                 " + std::to_string(conf::matrixBlockSize) + ",\n";
    metricsJSON << "\t \"workGroupSize\":                   " + std::to_string(conf::workGroupSize) + ",\n";
    metricsJSON << "\t \"workGroupSizeVector\":             " + std::to_string(conf::workGroupSizeVector) + ",\n";
    metricsJSON << "\t \"workGroupSizeFinalScalarProduct\": " + std::to_string(conf::workGroupSizeFinalScalarProduct) + ",\n";
    metricsJSON << "\t \"iMax\":                            " + std::to_string(conf::iMax) + ",\n";
    metricsJSON << "\t \"epsilon\":                         " + std::to_string(conf::epsilon) + ",\n";
    metricsJSON << "\t \"updateInterval\":                  " + std::to_string(conf::updateInterval) + ",\n";
    metricsJSON << "\t \"initialProportionGPU\":            " + std::to_string(conf::initialProportionGPU) + ",\n";
    metricsJSON << "\t \"runtimeLBFactorCPU\":              " + std::to_string(conf::runtimeLBFactorCPU) + ",\n";
    metricsJSON << "\t \"blockUpdateThreshold\":            " + std::to_string(conf::blockUpdateThreshold) + ",\n";
    metricsJSON << "\t \"gpuOptimizationLevel\":            " + std::to_string(conf::gpuOptimizationLevel) + ",\n";
    metricsJSON << "\t \"cpuOptimizationLevel\":            " + std::to_string(conf::cpuOptimizationLevel) + ",\n";
    metricsJSON << "\t \"enableHWS\":                       " + std::to_string(conf::enableHWS) + ",\n";
    std::string omp_proc_bind_env = getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "";
    std::string omp_cores_env = getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "\"\"";
    metricsJSON << "\t \"cpuBind\":                         " + std::string("\"") + omp_proc_bind_env + "\"" + ",\n";
    metricsJSON << "\t \"ompCores\":                        " + omp_cores_env + ",\n";
    metricsJSON << std::string("\t \"mode\":                            ") + "\"" + conf::mode + "\"" + "\n";


    metricsJSON << "},\n";


    metricsJSON << "\"runtime-metrics\": {\n";

    metricsJSON << "\t \"iterationTimes\":         " + vectorToJSONString<double>(iterationTimes) + ",\n";

    metricsJSON << "\t \"memoryInitTime\":         " + std::to_string(memoryInitTime) + ",\n";
    metricsJSON << "\t \"resultCopyTime\":         " + std::to_string(resultCopyTime) + ",\n";
    metricsJSON << "\t \"totalTime\":              " + std::to_string(totalTime) + ",\n";
    metricsJSON << "\t \"choleskySolveStepTime\":  " + std::to_string(solveTime) + ",\n";

    metricsJSON << "\t \"averageUtilization_GPU\": " + vectorToJSONString<double>(averageUtilization_GPU) + ",\n";
    metricsJSON << "\t \"averageUtilization_CPU\": " + vectorToJSONString<double>(averageUtilization_CPU) + ",\n";

    metricsJSON << "\t \"powerDraw_GPU\":          " + vectorToJSONString<double>(powerDraw_GPU) + ",\n";
    metricsJSON << "\t \"powerDraw_CPU\":          " + vectorToJSONString<double>(powerDraw_CPU) + ",\n";

    metricsJSON << "\t \"blockCounts_GPU\":        " + vectorToJSONString<std::size_t>(blockCounts_GPU) + ",\n";
    metricsJSON << "\t \"blockCounts_CPU\":        " + vectorToJSONString<std::size_t>(blockCounts_CPU) + ",\n";

    if (conf::algorithm == "cg") {
        metricsJSON << "\t \"matrixVectorTimes_GPU\":  " + vectorToJSONString<double>(matrixVectorTimes_GPU) + ",\n";
        metricsJSON << "\t \"matrixVectorTimes_CPU\":  " + vectorToJSONString<double>(matrixVectorTimes_CPU) + ",\n";

        metricsJSON << "\t \"times_q\":                " + vectorToJSONString<double>(times_q) + ",\n";
        metricsJSON << "\t \"times_alpha\":            " + vectorToJSONString<double>(times_alpha) + ",\n";
        metricsJSON << "\t \"times_x\":                " + vectorToJSONString<double>(times_x) + ",\n";
        metricsJSON << "\t \"times_r\":                " + vectorToJSONString<double>(times_r) + ",\n";
        metricsJSON << "\t \"times_delta\":            " + vectorToJSONString<double>(times_delta) + ",\n";
        metricsJSON << "\t \"times_d\":                " + vectorToJSONString<double>(times_d) + ",\n";

        metricsJSON << "\t \"memcopy_d\":              " + vectorToJSONString<double>(memcopy_d) + ",\n";
    } else if (conf::algorithm == "cholesky") {
        metricsJSON << "\t \"shiftTimes\":                      " + vectorToJSONString<double>(shiftTimes) + ",\n";
        metricsJSON << "\t \"choleskyDiagonalBlockTimes\":      " + vectorToJSONString<double>(choleskyDiagonalBlockTimes) + ",\n";
        metricsJSON << "\t \"copyTimes\":                       " + vectorToJSONString<double>(copyTimes) + ",\n";

        metricsJSON << "\t \"triangularSolveTimes_GPU\":        " + vectorToJSONString<double>(triangularSolveTimes_GPU) + ",\n";
        metricsJSON << "\t \"triangularSolveTimes_CPU\":        " + vectorToJSONString<double>(triangularSolveTimes_CPU) + ",\n";
        metricsJSON << "\t \"triangularSolveTimes_total\":      " + vectorToJSONString<double>(triangularSolveTimes_total) + ",\n";

        metricsJSON << "\t \"matrixMatrixDiagonalTimes_GPU\":   " + vectorToJSONString<double>(matrixMatrixDiagonalTimes_GPU) + ",\n";
        metricsJSON << "\t \"matrixMatrixDiagonalTimes_CPU\":   " + vectorToJSONString<double>(matrixMatrixDiagonalTimes_CPU) + ",\n";
        metricsJSON << "\t \"matrixMatrixDiagonalTimes_total\": " + vectorToJSONString<double>(matrixMatrixDiagonalTimes_total) + ",\n";

        metricsJSON << "\t \"matrixMatrixTimes_GPU\":           " + vectorToJSONString<double>(matrixMatrixTimes_GPU) + ",\n";
        metricsJSON << "\t \"matrixMatrixTimes_CPU\":           " + vectorToJSONString<double>(matrixMatrixTimes_CPU) + ",\n";
        metricsJSON << "\t \"matrixMatrixTimes_total\":         " + vectorToJSONString<double>(matrixMatrixTimes_total) + ",\n";
    }


#ifndef INTEL
    metricsJSON << "\t \"rawUtilizationData_GPU\": " + vectorToJSONString<unsigned int>(generalSamples_GPU.get_compute_utilization().value_or(std::vector<unsigned int>(0))) + ",\n";
#else
    metricsJSON << "\t \"rawUtilizationData_GPU\": " + std::string("[]") + ",\n";
#endif
    metricsJSON << "\t \"rawUtilizationData_CPU\": " + vectorToJSONString<double>(generalSamples_CPU.get_compute_utilization().value_or(std::vector<double>(0))) + ",\n";

    metricsJSON << "\t \"rawPowerData_GPU\":       " + vectorToJSONString<double>(powerSamples_GPU.get_power_usage().value_or(std::vector<double>(0))) + ",\n";
    metricsJSON << "\t \"rawPowerData_CPU\":       " + vectorToJSONString<double>(powerSamples_CPU.get_power_usage().value_or(std::vector<double>(0))) + ",\n";

    metricsJSON << "\t \"rawEnergyData_GPU\":      " + vectorToJSONString<double>(powerSamples_GPU.get_power_total_energy_consumption().value_or(std::vector<double>(0))) + ",\n";
    metricsJSON << "\t \"rawEnergyData_CPU\":      " + vectorToJSONString<double>(powerSamples_CPU.get_power_total_energy_consumption().value_or(std::vector<double>(0))) + ",\n";

    if (conf::advancedSampling) {
        metricsJSON << "\t \"rawClockData_CPU\":       " + vectorToJSONString<unsigned int>(cpu_sampler->clock_samples().get_clock_frequency().value_or(std::vector<unsigned int>(0))) + ",\n";
        metricsJSON << "\t \"rawTempData_CPU\":        " + vectorToJSONString<double>(cpu_sampler->temperature_samples().get_temperature().value_or(std::vector<double>(0))) + ",\n";

        metricsJSON << "\t \"rawClockData_GPU\":       " + vectorToJSONString<double>(gpu_sampler->clock_samples().get_clock_frequency().value_or(std::vector<double>(0))) + ",\n";
        metricsJSON << "\t \"rawTempData_GPU\":        " + vectorToJSONString<double>(gpu_sampler->temperature_samples().get_temperature().value_or(std::vector<double>(0))) + ",\n";
    }


    std::vector<long> timePointsGPU_general;
    for (auto& x : gpu_sampler->sampling_time_points()) {
        timePointsGPU_general.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\t \"timePointsGPU\":  " + vectorToJSONString<long>(timePointsGPU_general) + ",\n";

    std::vector<long> timePointsCPU_general;
    for (auto& x : cpu_sampler->sampling_time_points()) {
        timePointsCPU_general.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\t \"timePointsCPU\":  " + vectorToJSONString<long>(timePointsCPU_general) + "\n";

    metricsJSON << "}\n";

    metricsJSON << "}\n";
}

template <typename T>
std::string MetricsTracker::vectorToJSONString(std::vector<T> vector) {
    std::string jsonString;
    if (vector.empty()) {
        jsonString = "[]";
        return jsonString;
    }

    jsonString += "[";
    for (unsigned int i = 0; i < vector.size() - 1; ++i) {
        jsonString += std::to_string(vector[i]);
        jsonString += ", ";
    }
    jsonString += std::to_string(vector[vector.size() - 1]);
    jsonString += "]";

    return jsonString;
}
