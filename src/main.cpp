#include <iostream>
#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>
#include <cxxopts.hpp>

#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"
#include "CG.hpp"
#include "../cmake-build-release/_deps/hws-src/include/hws/cpu/hardware_sampler.hpp"

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
        ("d,device", "Specifies which devices the solver will use. Has to be 'cpu', 'gpu' or 'mixed'.",
         cxxopts::value<std::string>());

    const auto arguments = argumentOptions.parse(argc, argv);

    std::string path_A;
    std::string path_b;
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

    // hws::system_hardware_sampler sampler{hws::sample_category::idle_state};
    //
    // sampler.start_sampling();

    queue gpuQueue(gpu_selector_v);
    queue cpuQueue(cpu_selector_v);

    std::cout << "GPU: " << gpuQueue.get_device().get_info<info::device::name>() << std::endl;
    std::cout << "CPU: " << cpuQueue.get_device().get_info<info::device::name>() << std::endl;

    CG algorithm(path_A, path_b, cpuQueue, gpuQueue);
    // algorithm.solveHeterogeneous_static();
    // algorithm.solve_GPU();
    algorithm.solve_CPU();
    // sleep(3);





    // sampler.stop_sampling();
    // sampler.dump_yaml("test.yaml");

    // auto* cpu_sampler =  dynamic_cast<hws::cpu_hardware_sampler*>(sampler.samplers()[0].get());
    // hws::cpu_power_samples power_samples = cpu_sampler->power_samples();
    // auto power = power_samples.get_power_total_energy_consumption().value_or(std::vector<double>(1));
    //
    // std::cout << power[power.size() - 1] << std::endl;

    return 0;
}
