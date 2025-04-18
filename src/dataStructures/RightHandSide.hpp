#ifndef RIGHTHANDSIDE_HPP
#define RIGHTHANDSIDE_HPP

#include <vector>
#include <sycl/sycl.hpp>

#include "Configuration.hpp"

class RightHandSide {
public:
    /**
     * Constructor of the class.
     * Automatically resizes the vector rightHandSideData to the correct size.
     *
     * @param N Dimension N of the Nx1 right-hand side
     * @param blockSize block size of the right hand side equal to the block size of the matrix
     * @param queue SYCL queue for allocating memory
     */
    RightHandSide(std::size_t N, int blockSize, sycl::queue& queue);

    const std::size_t N; /// Size N of the Nx1 right hand size
    const int blockSize; /// The right hand side can be partitioned in blockSize blocks
    const int blockCountX; /// block Count in X direction

    /// internal data structure using SYCL host memory
    std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>> rightHandSideData;
};


#endif //RIGHTHANDSIDE_HPP
