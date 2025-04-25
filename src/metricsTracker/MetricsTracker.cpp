#include <numeric>
#include <iostream>
#include <filesystem>

#include "MetricsTracker.hpp"

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
        auto *cpu_sampler = dynamic_cast<hws::cpu_hardware_sampler *>(utilizationSampler.samplers()[0].get());
        auto *gpu_sampler = dynamic_cast<hws::gpu_nvidia_hardware_sampler *>(utilizationSampler.samplers()[1].get());
        auto *cpu_sampler_power = dynamic_cast<hws::cpu_hardware_sampler *>(powerSampler.samplers()[0].get());
        auto *gpu_sampler_power = dynamic_cast<hws::gpu_nvidia_hardware_sampler *>(powerSampler.samplers()[1].get());

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
            std::cout << "Average GPU util: " << averageUtil << std::endl;
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
            std::cout << "Average CPU util: " << averageUtil << std::endl;
        }

        if (powerSamples_CPU.get_power_total_energy_consumption().has_value()) {
            double powerDraw = 0.0;
            if (nextTimePointPower_CPU < powerSamples_CPU.get_power_total_energy_consumption().value().size() - 1) {
                // at least 2 new power samples are available
                powerDraw = powerSamples_CPU.get_power_total_energy_consumption().value().back() -
                            powerSamples_CPU.get_power_total_energy_consumption().value()[nextTimePointPower_CPU];
            } else {
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!\n";
            }
            nextTimePointPower_CPU = powerSamples_CPU.get_power_total_energy_consumption().value().size();
            powerDraw_CPU.push_back(powerDraw);
            std::cout << "CPU Power draw " << powerDraw << std::endl;
        }

        if (powerSamples_GPU.get_power_total_energy_consumption().has_value()) {
            double powerDraw = 0.0;
            if (nextTimePointPower_GPU < powerSamples_GPU.get_power_total_energy_consumption().value().size() - 1) {
                // at least 2 new power samples are available
                powerDraw = powerSamples_GPU.get_power_total_energy_consumption().value().back() -
                            powerSamples_GPU.get_power_total_energy_consumption().value()[nextTimePointPower_GPU];
            } else {
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!\n";
            }
            nextTimePointPower_GPU = powerSamples_GPU.get_power_total_energy_consumption().value().size();
            powerDraw_GPU.push_back(powerDraw);
            std::cout << "GPU Power draw " << powerDraw << std::endl;
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

void MetricsTracker::writeJSON(std::string &path) {

    auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto timeStruct = *std::localtime(&currentTime);

    std::ostringstream timeStringStream;
    timeStringStream << std::put_time(&timeStruct, "%Y_%m_%d_%H-%M-%S");
    std::string timeString = timeStringStream.str();
    std::string filePath = path + "/" + timeString;
    std::filesystem::create_directories(filePath);

    std::ofstream metricsJSON(filePath + "/metrics.json");

    auto *cpu_sampler = dynamic_cast<hws::cpu_hardware_sampler *>(utilizationSampler.samplers()[0].get());
    auto *gpu_sampler = dynamic_cast<hws::gpu_nvidia_hardware_sampler *>(utilizationSampler.samplers()[1].get());
    auto *cpu_sampler_power = dynamic_cast<hws::cpu_hardware_sampler *>(powerSampler.samplers()[0].get());
    auto *gpu_sampler_power = dynamic_cast<hws::gpu_nvidia_hardware_sampler *>(powerSampler.samplers()[1].get());

    hws::cpu_general_samples generalSamples_CPU = cpu_sampler->general_samples();
    hws::nvml_general_samples generalSamples_GPU = gpu_sampler->general_samples();
    hws::cpu_power_samples powerSamples_CPU = cpu_sampler_power->power_samples();
    hws::nvml_power_samples powerSamples_GPU = gpu_sampler_power->power_samples();

    metricsJSON << "{\n";

    metricsJSON << "\"iterationTimes\":" + vectorToJSONString<double>(iterationTimes) + ",\n";

    metricsJSON << "\"averageUtilization_GPU\":" + vectorToJSONString<double>(averageUtilization_GPU) + ",\n";
    metricsJSON << "\"averageUtilization_CPU\":" + vectorToJSONString<double>(averageUtilization_CPU) + ",\n";

    metricsJSON << "\"powerDraw_GPU\":" + vectorToJSONString<double>(powerDraw_GPU) + ",\n";
    metricsJSON << "\"powerDraw_CPU\":" + vectorToJSONString<double>(powerDraw_CPU) + ",\n";

    metricsJSON << "\"blockCounts_GPU\":" + vectorToJSONString<std::size_t>(blockCounts_GPU) + ",\n";
    metricsJSON << "\"blockCounts_CPU\":" + vectorToJSONString<std::size_t>(blockCounts_CPU) + ",\n";

    metricsJSON << "\"matrixVectorTimes_GPU\":" + vectorToJSONString<double>(matrixVectorTimes_GPU) + ",\n";
    metricsJSON << "\"matrixVectorTimes_CPU\":" + vectorToJSONString<double>(matrixVectorTimes_CPU) + ",\n";

    metricsJSON << "\"times_q\":" + vectorToJSONString<double>(times_q) + ",\n";
    metricsJSON << "\"times_alpha\":" + vectorToJSONString<double>(times_alpha) + ",\n";
    metricsJSON << "\"times_x\":" + vectorToJSONString<double>(times_x) + ",\n";
    metricsJSON << "\"times_r\":" + vectorToJSONString<double>(times_r) + ",\n";
    metricsJSON << "\"times_delta\":" + vectorToJSONString<double>(times_delta) + ",\n";
    metricsJSON << "\"times_d\":" + vectorToJSONString<double>(times_d) + ",\n";

    metricsJSON << "\"memcopy_d\":" + vectorToJSONString<double>(memcopy_d) + ",\n";

    metricsJSON << "\"rawUtilizationData_GPU\":" + vectorToJSONString<unsigned int>(
            generalSamples_GPU.get_compute_utilization().value_or(std::vector<unsigned int>(0))) + ",\n";
    metricsJSON << "\"rawUtilizationData_CPU\":" + vectorToJSONString<double>(
            generalSamples_CPU.get_compute_utilization().value_or(std::vector<double>(0))) + ",\n";

    metricsJSON << "\"rawPowerData_GPU\":" + vectorToJSONString<double>(
            powerSamples_GPU.get_power_usage().value_or(std::vector<double>(0))) + ",\n";
    metricsJSON << "\"rawPowerData_CPU\":" + vectorToJSONString<double>(
            powerSamples_CPU.get_power_usage().value_or(std::vector<double>(0))) + ",\n";

    metricsJSON << "\"rawEnergyData_GPU\":" + vectorToJSONString<double>(
            powerSamples_GPU.get_power_total_energy_consumption().value_or(std::vector<double>(0))) + ",\n";
    metricsJSON << "\"rawEnergyData_CPU\":" + vectorToJSONString<double>(
            powerSamples_CPU.get_power_total_energy_consumption().value_or(std::vector<double>(0))) + ",\n";

    std::vector<long> timePointsGPU_general;
    for (auto &x: gpu_sampler->sampling_time_points()) {
        timePointsGPU_general.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\"timePointsGPU_general\":" + vectorToJSONString<long>(timePointsGPU_general) + ",\n";

    std::vector<long> timePointsCPU_general;
    for (auto &x: cpu_sampler->sampling_time_points()) {
        timePointsCPU_general.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\"timePointsCPU_general\":" + vectorToJSONString<long>(timePointsCPU_general) + ",\n";

    std::vector<long> timePointsGPU_power;
    for (auto &x: gpu_sampler_power->sampling_time_points()) {
        timePointsGPU_power.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\"timePointsGPU_power\":" + vectorToJSONString<long>(timePointsGPU_power) + ",\n";

    std::vector<long> timePointsCPU_power;
    for (auto &x: cpu_sampler_power->sampling_time_points()) {
        timePointsCPU_power.push_back(x.time_since_epoch().count());
    }
    metricsJSON << "\"timePointsCPU_power\":" + vectorToJSONString<long>(timePointsCPU_power) + "\n";

    metricsJSON << "}\n";


}

template<typename T>
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
