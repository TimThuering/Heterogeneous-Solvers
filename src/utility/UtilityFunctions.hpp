#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP

#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include "Configuration.hpp"
#include "RightHandSide.hpp"

/**
 * This class contains various utility functions
 */
class UtilityFunctions {
public:
    /**
     * This operation writes the result vector x to a txt file
     * @param path result path
     * @param x result x
     */
    static void writeResult(const std::string& path, const std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>>& x);

    /**
     * This operation returns a string with the current date and time.
     *
     * @return string with the current date and time
     */
    static std::string getTimeString();

    /**
     * This operation measures the idle power draw using the hws library
     */
    static void measureIdlePowerCPU();

    /**
     * Validates the result of the Cholesky decomposition and the solve step by computing the average error of Ax - b.
     *
     * @param b right hand side
     * @param cpuQueue CPU queue
     * @param gpuQueue GPU queue
     * @param path_gp_input Gaussian process input data
     * @param path_gp_output Gaussian process output data
     * @return average error of Ax - b
     */
    static double checkResult(RightHandSide& b, sycl::queue cpuQueue, sycl::queue gpuQueue, std::string path_gp_input, std::string path_gp_output);


};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP
