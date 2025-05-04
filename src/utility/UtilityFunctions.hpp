#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP

#include <string>
#include <vector>
#include <sycl/sycl.hpp>


class UtilityFunctions {
public:
    static void writeResult(const std::string& path, const std::vector<double, sycl::usm_allocator<double, sycl::usm::alloc::shared>>& x);

    static std::string getTimeString();


};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_UTILITYFUNCTIONS_HPP
