#include <numeric>
#include <iostream>
#include <filesystem>

#include "MetricsTracker.hpp"
#include "UtilityFunctions.hpp"
#include "Configuration.hpp"

void MetricsTracker::updateMetrics(std::size_t iteration, std::size_t blockCount_GPU, std::size_t blockCount_CPU,
                                   double iterationTime,
                                   int updateInterval) {
    utilizationSampler.pause_sampling();
    powerSampler.pause_sampling();

    // add new block count of iteration
    blockCounts_GPU.push_back(blockCount_GPU);
    blockCounts_CPU.push_back(blockCount_CPU);

    // add time of iteration
    iterationTimes.push_back(iterationTime);

    // track metrics for load balancing before every update interval
    if ((iteration + 1) % updateInterval == 0) {
        // get samples for power and utilization from the hws library
        auto* cpu_sampler = dynamic_cast<hws::cpu_hardware_sampler*>(utilizationSampler.samplers()[0].get());
        auto* gpu_sampler = dynamic_cast<hws::gpu_nvidia_hardware_sampler*>(utilizationSampler.samplers()[1].get());
        auto* cpu_sampler_power = dynamic_cast<hws::cpu_hardware_sampler*>(powerSampler.samplers()[0].get());
        auto* gpu_sampler_power = dynamic_cast<hws::gpu_nvidia_hardware_sampler*>(powerSampler.samplers()[1].get());

        hws::cpu_general_samples generalSamples_CPU = cpu_sampler->general_samples();
        hws::nvml_general_samples generalSamples_GPU = gpu_sampler->general_samples();
        hws::cpu_power_samples powerSamples_CPU = cpu_sampler_power->power_samples();
        hws::nvml_power_samples powerSamples_GPU = gpu_sampler_power->power_samples();

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
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!" << std::endl;
                averageUtil = generalSamples_GPU.get_compute_utilization().value().back();
            }
            // update index where the new samples will begin for the next interval
            nextTimePoint_GPU = generalSamples_GPU.get_compute_utilization().value().size();
            averageUtilization_GPU.push_back(averageUtil);
            // std::cout << "Average GPU util: " << averageUtil << std::endl;
        }

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
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!" << std::endl;
                averageUtil = generalSamples_CPU.get_compute_utilization().value().back();
            }
            nextTimePoint_CPU = generalSamples_CPU.get_compute_utilization().value().size();
            averageUtilization_CPU.push_back(averageUtil);
            // std::cout << "Average CPU util: " << averageUtil << std::endl;
        }

        if (powerSamples_CPU.get_power_usage().has_value()) {
            double powerDraw = 0.0;
            if (nextTimePointPower_CPU < powerSamples_CPU.get_power_usage().value().size()) {
                // at least 1 new power samples is available
                std::vector<double> CPU_watts(
                    powerSamples_CPU.get_power_usage().value().begin() + static_cast<long>(nextTimePointPower_CPU),
                    powerSamples_CPU.get_power_usage().value().end());

                std::vector<double> CPU_mvTimes(matrixVectorTimes_CPU.end() - updateInterval,
                                                matrixVectorTimes_CPU.end());
                double averageMVTime = std::accumulate(CPU_mvTimes.begin(), CPU_mvTimes.end(), 0.0) /
                    static_cast<double>(CPU_mvTimes.size());

                powerDraw = std::accumulate(CPU_watts.begin(), CPU_watts.end(), 0.0) /
                    static_cast<double>(CPU_watts.size());
                powerDraw = powerDraw * averageMVTime;
            } else {
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!\n";
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

                std::vector<double> GPU_mvTimes(matrixVectorTimes_GPU.end() - updateInterval,
                                                matrixVectorTimes_GPU.end());
                double averageMVTime = std::accumulate(GPU_mvTimes.begin(), GPU_mvTimes.end(), 0.0) /
                    static_cast<double>(GPU_mvTimes.size());
                powerDraw =
                    std::accumulate(GPU_watts.begin(), GPU_watts.end(), 0.0) / static_cast<double>(GPU_watts.size());
                powerDraw = powerDraw * averageMVTime;
            } else {
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!\n";
                powerDraw = powerSamples_GPU.get_power_usage().value().back();
            }
            nextTimePointPower_GPU = powerSamples_GPU.get_power_usage().value().size();
            powerDraw_GPU.push_back(powerDraw);
        }
    }

    utilizationSampler.resume_sampling();
    powerSampler.resume_sampling();
}

void MetricsTracker::startTracking() {
    utilizationSampler.start_sampling();
    powerSampler.start_sampling();
}

void MetricsTracker::endTracking() {
    utilizationSampler.stop_sampling();
    powerSampler.stop_sampling();
}

void MetricsTracker::writeJSON(std::string& path) {
    std::ofstream metricsJSON(path + "/metrics.json");

    auto* cpu_sampler = dynamic_cast<hws::cpu_hardware_sampler*>(utilizationSampler.samplers()[0].get());
    auto* gpu_sampler = dynamic_cast<hws::gpu_nvidia_hardware_sampler*>(utilizationSampler.samplers()[1].get());
    auto* cpu_sampler_power = dynamic_cast<hws::cpu_hardware_sampler*>(powerSampler.samplers()[0].get());
    auto* gpu_sampler_power = dynamic_cast<hws::gpu_nvidia_hardware_sampler*>(powerSampler.samplers()[1].get());

    hws::cpu_general_samples generalSamples_CPU = cpu_sampler->general_samples();
    hws::nvml_general_samples generalSamples_GPU = gpu_sampler->general_samples();
    hws::cpu_power_samples powerSamples_CPU = cpu_sampler_power->power_samples();
    hws::nvml_power_samples powerSamples_GPU = gpu_sampler_power->power_samples();

    metricsJSON << "{\n";

    metricsJSON << "\"configuration\": {\n";
    if (cpu_sampler->general_samples().get_name().has_value()) {
        metricsJSON << std::string("\t \"CPU\":                             ") + "\"" + cpu_sampler->general_samples().
            get_name().value() + "\"" + ",\n";
    }
    if (gpu_sampler->general_samples().get_name().has_value()) {
        metricsJSON << std::string("\t \"GPU\":                             ") + "\"" + gpu_sampler->general_samples().
            get_name().value() + "\"" + ",\n";
    }

    metricsJSON << "\t \"N\":                               " + std::to_string(conf::N) + ",\n";

    metricsJSON << "\t \"matrixBlockSize\":                 " + std::to_string(conf::matrixBlockSize) + ",\n";
    metricsJSON << "\t \"workGroupSize\":                   " + std::to_string(conf::workGroupSize) + ",\n";
    metricsJSON << "\t \"workGroupSizeVector\":             " + std::to_string(conf::workGroupSizeVector) + ",\n";
    metricsJSON << "\t \"workGroupSizeFinalScalarProduct\": " + std::to_string(conf::workGroupSizeFinalScalarProduct) +
        ",\n";
    metricsJSON << "\t \"iMax\":                            " + std::to_string(conf::iMax) + ",\n";
    metricsJSON << "\t \"epsilon\":                         " + std::to_string(conf::epsilon) + ",\n";
    metricsJSON << "\t \"updateInterval\":                  " + std::to_string(conf::updateInterval) + ",\n";
    metricsJSON << "\t \"initialProportionGPU\":            " + std::to_string(conf::initialProportionGPU) + ",\n";
    metricsJSON << "\t \"runtimeLBFactorCPU\":              " + std::to_string(conf::runtimeLBFactorCPU) + ",\n";
    metricsJSON << "\t \"blockUpdateThreshold\":            " + std::to_string(conf::blockUpdateThreshold) + ",\n";
    metricsJSON << std::string("\t \"mode\":                            ") + "\"" + conf::mode + "\"" + "\n";


    metricsJSON << "},\n";


    metricsJSON << "\"runtime-metrics\": {\n";

    metricsJSON << "\t \"iterationTimes\":         " + vectorToJSONString<double>(iterationTimes) + ",\n";

    metricsJSON << "\t \"averageUtilization_GPU\": " + vectorToJSONString<double>(averageUtilization_GPU) + ",\n";
    metricsJSON << "\t \"averageUtilization_CPU\": " + vectorToJSONString<double>(averageUtilization_CPU) + ",\n";

    metricsJSON << "\t \"powerDraw_GPU\":          " + vectorToJSONString<double>(powerDraw_GPU) + ",\n";
    metricsJSON << "\t \"powerDraw_CPU\":          " + vectorToJSONString<double>(powerDraw_CPU) + ",\n";

    metricsJSON << "\t \"blockCounts_GPU\":        " + vectorToJSONString<std::size_t>(blockCounts_GPU) + ",\n";
    metricsJSON << "\t \"blockCounts_CPU\":        " + vectorToJSONString<std::size_t>(blockCounts_CPU) + ",\n";

    metricsJSON << "\t \"matrixVectorTimes_GPU\":  " + vectorToJSONString<double>(matrixVectorTimes_GPU) + ",\n";
    metricsJSON << "\t \"matrixVectorTimes_CPU\":  " + vectorToJSONString<double>(matrixVectorTimes_CPU) + ",\n";

    metricsJSON << "\t \"times_q\":                " + vectorToJSONString<double>(times_q) + ",\n";
    metricsJSON << "\t \"times_alpha\":            " + vectorToJSONString<double>(times_alpha) + ",\n";
    metricsJSON << "\t \"times_x\":                " + vectorToJSONString<double>(times_x) + ",\n";
    metricsJSON << "\t \"times_r\":                " + vectorToJSONString<double>(times_r) + ",\n";
    metricsJSON << "\t \"times_delta\":            " + vectorToJSONString<double>(times_delta) + ",\n";
    metricsJSON << "\t \"times_d\":                " + vectorToJSONString<double>(times_d) + ",\n";

    metricsJSON << "\t \"memcopy_d\":              " + vectorToJSONString<double>(memcopy_d) + ",\n";

    metricsJSON << "\t \"rawUtilizationData_GPU\": " + vectorToJSONString<unsigned int>(
        generalSamples_GPU.get_compute_utilization().value_or(std::vector<unsigned int>(0))) + ",\n";
    metricsJSON << "\t \"rawUtilizationData_CPU\": " + vectorToJSONString<double>(
        generalSamples_CPU.get_compute_utilization().value_or(std::vector<double>(0))) + ",\n";

    metricsJSON << "\t \"rawPowerData_GPU\":       " + vectorToJSONString<double>(
        powerSamples_GPU.get_power_usage().value_or(std::vector<double>(0))) + ",\n";
    metricsJSON << "\t \"rawPowerData_CPU\":       " + vectorToJSONString<double>(
        powerSamples_CPU.get_power_usage().value_or(std::vector<double>(0))) + ",\n";

    metricsJSON << "\t \"rawEnergyData_GPU\":      " + vectorToJSONString<double>(
        powerSamples_GPU.get_power_total_energy_consumption().value_or(std::vector<double>(0))) + ",\n";
    metricsJSON << "\t \"rawEnergyData_CPU\":      " + vectorToJSONString<double>(
        powerSamples_CPU.get_power_total_energy_consumption().value_or(std::vector<double>(0))) + ",\n";

    std::vector<long> timePointsGPU_general;
    for (auto& x : gpu_sampler->sampling_time_points()) {
        timePointsGPU_general.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\t \"timePointsGPU_general\":  " + vectorToJSONString<long>(timePointsGPU_general) + ",\n";

    std::vector<long> timePointsCPU_general;
    for (auto& x : cpu_sampler->sampling_time_points()) {
        timePointsCPU_general.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\t \"timePointsCPU_general\":  " + vectorToJSONString<long>(timePointsCPU_general) + ",\n";

    std::vector<long> timePointsGPU_power;
    for (auto& x : gpu_sampler_power->sampling_time_points()) {
        timePointsGPU_power.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\t \"timePointsGPU_power\":    " + vectorToJSONString<long>(timePointsGPU_power) + ",\n";

    std::vector<long> timePointsCPU_power;
    for (auto& x : cpu_sampler_power->sampling_time_points()) {
        timePointsCPU_power.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\t \"timePointsCPU_power\":    " + vectorToJSONString<long>(timePointsCPU_power) + "\n";

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
