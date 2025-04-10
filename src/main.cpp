#include <iostream>
#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>

#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"
#include "CG.hpp"

using namespace sycl;

int main()
{
#ifdef USE_DOUBLE
    std::cout << "Using FP64 double precision" << std::endl;
#else
    std::cout << "Using FP32 single precision" << std::endl;
#endif

    std::string path_A = "../matrixGenerator/A_20.txt";
    std::string path_b = "../matrixGenerator/b_20.txt";
    // std::string path2 = "../matrixGenerator/matrixBlocked.txt";

    queue gpuQueue(gpu_selector_v);
    queue cpuQueue(cpu_selector_v);

    CG algorithm(path_A, path_b, cpuQueue, gpuQueue);
    algorithm.solve();


    // sycl::queue gpuQueue(sycl::gpu_selector_v);
    // sycl::queue cpuQueue(sycl::cpu_selector_v);
    //
    // hws::system_hardware_sampler sampler{hws::sample_category::power};
    //
    // sampler.start_sampling();
    //
    // std::cout << "GPU: " << gpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;
    //
    // std::cout << "CPU: " << cpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;
    //
    //
    // sleep(3);
    //
    // sampler.stop_sampling();
    // sampler.dump_yaml("test.yaml");

    return 0;
}
