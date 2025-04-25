#include <iostream>
#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>
#include <hws/cpu/hardware_sampler.hpp>
#include <hws/gpu_nvidia/hardware_sampler.hpp>
#include <hws/gpu_amd/hardware_sampler.hpp>
#include <cxxopts.hpp>

#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"
#include "CG.hpp"
#include "LoadBalancer.hpp"
#include "StaticLoadBalancer.hpp"
#include "UtilizationLoadBalancer.hpp"
#include "RuntimeLoadBalancer.hpp"

using namespace sycl;

int main(int argc, char* argv[]) {
#ifdef USE_DOUBLE
    std::cout << "Using FP64 double precision" << std::endl;
#else
    std::cout << "Using FP32 single precision" << std::endl;
#endif


    cxxopts::Options argumentOptions("Heterogeneous Conjugate Gradients", "CG Algorithm with CPU-GPU co-execution");

    argumentOptions.add_options()
        ("A,path_A", "path to .txt file containing symmetric positive definite matrix A",
         cxxopts::value<std::string>())
        ("b,path_b", "path to .txt file containing the right-hand side b", cxxopts::value<std::string>())
        ("o,output", "path to the output directory", cxxopts::value<std::string>())
        ("m,mode", "Specifies the load balancing mode between CPU and GPU. Has to be 'static', 'runtime', 'power' or 'util'.",
         cxxopts::value<std::string>());

    const auto arguments = argumentOptions.parse(argc, argv);

    std::string path_A;
    std::string path_b;
    std::string mode;
    if (arguments.count("path_A")) {
        path_A = arguments["path_A"].as<std::string>();
    } else {
        throw std::runtime_error("No path to .txt file for matrix A specified");
    }

    if (arguments.count("path_b")) {
        path_b = arguments["path_b"].as<std::string>();
    } else {
        throw std::runtime_error("No path to .txt file for right-hand side b specified");
    }

    if (arguments.count("output")) {
        conf::outputPath = arguments["output"].as<std::string>();
    }

    if (arguments.count("mode")) {
        mode = arguments["mode"].as<std::string>();
    } else {
        mode = "runtime";
    }


    queue gpuQueue(gpu_selector_v, sycl::property::queue::enable_profiling());
    queue cpuQueue(cpu_selector_v, sycl::property::queue::enable_profiling());

    std::cout << "GPU: " << gpuQueue.get_device().get_info<info::device::name>() << std::endl;
    std::cout << "CPU: " << cpuQueue.get_device().get_info<info::device::name>() << std::endl;

    std::shared_ptr<LoadBalancer> loadBalancer;
    if (mode == "static") {
        loadBalancer = std::make_shared<StaticLoadBalancer>(conf::updateInterval,conf::initialProportionGPU);
    } else if (mode == "runtime") {
        loadBalancer = std::make_shared<RuntimeLoadBalancer>(conf::updateInterval,conf::initialProportionGPU);
    } else if (mode == "util") {
        loadBalancer = std::make_shared<UtilizationLoadBalancer>(conf::updateInterval,conf::initialProportionGPU);
    } else if (mode == "power") {
        throw std::runtime_error("Power load balancing not implemented yet");
    } else {
        throw std::runtime_error("Invalid mode selected: '" + mode + "' --> must be 'static', 'runtime', 'power' or 'util'");
    }


    CG algorithm(path_A, path_b, cpuQueue, gpuQueue, loadBalancer);

    algorithm.solveHeterogeneous();

    return 0;
}
