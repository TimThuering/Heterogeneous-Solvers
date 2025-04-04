#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
    sycl::queue gpuQueue(sycl::gpu_selector_v);
    sycl::queue cpuQueue(sycl::cpu_selector_v);

    std::cout << "GPU: " << gpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::cout << "CPU: " << cpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    return 0;
}
