#ifndef RIGHTHANDSIDE_HPP
#define RIGHTHANDSIDE_HPP

#include <vector>
#include <sycl/sycl.hpp>

#include "Configuration.hpp"

class RightHandSide
{
public:
    /**
     * Constructor of the class.
     * Automatically resizes the vector rightHandSideData to the correct size.
     */
    RightHandSide(std::size_t N, int blockSize, sycl::queue& queue);

    ~RightHandSide();

    const std::size_t N; /// Size N of the Nx1 right hand size
    const int blockSize; /// The right hand side can be partitioned in blockSize blocks
    const int blockCountX; /// block Count in X direction
    sycl::queue cpuQueue;

    // std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>> rightHandSideData;
     conf::fp_type* rightHandSideData; /// internal data structure allocated as SYCL host memory
};


#endif //RIGHTHANDSIDE_HPP
