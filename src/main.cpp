#include <iostream>
#include "matrixData/SymmetricMatrix.hpp"
#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>

int main()
{
    SymmetricMatrix matrix;
    matrix.example();

    sycl::queue gpuQueue(sycl::gpu_selector_v);
    sycl::queue cpuQueue(sycl::cpu_selector_v);

    hws::system_hardware_sampler sampler{hws::sample_category::power};

    sampler.start_sampling();

    std::cout << "GPU: " << gpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::cout << "CPU: " << cpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;


    sleep(3);
#ifdef USE_DOUBLE
    std::cout << "Using FP64 double precision" << std::endl;
#else
    std::cout << "Using FP32 single precision" << std::endl;
#endif



    sampler.stop_sampling();
    sampler.dump_yaml("test.yaml");

    return 0;
}
