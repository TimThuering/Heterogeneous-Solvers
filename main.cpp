#include <iostream>
#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>

int main()
{
    sycl::queue gpuQueue(sycl::gpu_selector_v);
    sycl::queue cpuQueue(sycl::cpu_selector_v);

    std::chrono::milliseconds interval = std::chrono::milliseconds(100);
    hws::system_hardware_sampler sampler{hws::sample_category::clock};

    sampler.start_sampling();

    std::cout << "GPU: " << gpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::cout << "CPU: " << cpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;


    sleep(10);


    sampler.stop_sampling();
    sampler.dump_yaml("test.yaml");
    return 0;
}
