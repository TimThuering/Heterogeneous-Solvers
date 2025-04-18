#include <sycl/sycl.hpp>

#include "RightHandSide.hpp"


RightHandSide::RightHandSide(const std::size_t N, const int blockSize, sycl::queue& queue) :
    N(N),
    blockSize(blockSize),
    blockCountX(std::ceil(static_cast<double>(N) / static_cast<double>(blockSize))),
    rightHandSideData(sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>(queue)) {
    // allocate memory for right-hand side storage
    rightHandSideData.resize(blockCountX * blockSize);
}
