#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP

#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include "Configuration.hpp"
#include "RightHandSide.hpp"


class UtilityFunctions {
public:
    static void writeResult(const std::string& path, const std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>>& x);

    static std::string getTimeString();

    static void measureIdlePowerCPU();

    static double checkResult(RightHandSide& b, sycl::queue cpuQueue, sycl::queue gpuQueue, std::string path_gp_input, std::string path_gp_output);


};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP
