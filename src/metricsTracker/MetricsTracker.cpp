#include <numeric>
#include <iostream>
#include "MetricsTracker.hpp"

void MetricsTracker::updateMetrics(std::size_t iteration, std::size_t blockCount_GPU, std::size_t blockCount_CPU,
                                   double iterationTime,
                                   int updateInterval) {
//    hws::event iterationEnd(std::chrono::steady_clock::now(), std::to_string(iteration));
//    utilizationSampler.add_event(iterationEnd);
//    powerSampler.add_event(iterationEnd);

    utilizationSampler.pause_sampling();
    powerSampler.pause_sampling();

    // add new block count of iteration
    blockCounts_GPU.push_back(blockCount_GPU);
    blockCounts_CPU.push_back(blockCount_CPU);

    // add time of iteration
    iterationTimes.push_back(iterationTime);

    if ((iteration + 1) % updateInterval == 0) {
        auto *cpu_sampler = dynamic_cast<hws::cpu_hardware_sampler *>(utilizationSampler.samplers()[0].get());
        auto *gpu_sampler = dynamic_cast<hws::gpu_nvidia_hardware_sampler *>(utilizationSampler.samplers()[1].get());

        hws::cpu_general_samples generalSamples_CPU = cpu_sampler->general_samples();
        hws::nvml_general_samples generalSamples_GPU = gpu_sampler->general_samples();

        if (generalSamples_GPU.get_compute_utilization().has_value()) {
            double averageUtil = 0.0;
            if (nextTimePoint_GPU < generalSamples_GPU.get_compute_utilization().value().size()) {
                std::vector<double> GPU_util(
                        generalSamples_GPU.get_compute_utilization().value().begin() + nextTimePoint_GPU,
                        generalSamples_GPU.get_compute_utilization().value().end());
                averageUtil =
                        std::accumulate(GPU_util.begin(), GPU_util.end(), 0.0) / static_cast<double>(GPU_util.size());
            } else {
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!" << std::endl;
                averageUtil = generalSamples_GPU.get_compute_utilization().value().back();
            }
            nextTimePoint_GPU = generalSamples_GPU.get_compute_utilization().value().size();
            averageUtilization_GPU.push_back(averageUtil);
            std::cout << "Average GPU util: " << averageUtil << std::endl;

        }

        if (generalSamples_CPU.get_compute_utilization().has_value()) {
            double averageUtil = 0.0;
            if (nextTimePoint_CPU < generalSamples_CPU.get_compute_utilization().value().size()) {
                std::vector<double> CPU_util(
                        generalSamples_CPU.get_compute_utilization().value().begin() + nextTimePoint_CPU,
                        generalSamples_CPU.get_compute_utilization().value().end());
                averageUtil =
                        std::accumulate(CPU_util.begin(), CPU_util.end(), 0.0) / static_cast<double>(CPU_util.size());
            } else {
                std::cerr << "\033[93m[WARNING]\033[0m Sampling frequency is too low!\n";
                averageUtil = generalSamples_CPU.get_compute_utilization().value().back();
            }
            nextTimePoint_CPU = generalSamples_CPU.get_compute_utilization().value().size();
            averageUtilization_CPU.push_back(averageUtil);
            std::cout << "Average CPU util: " << averageUtil << std::endl;
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
