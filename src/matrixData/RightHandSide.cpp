#include <sycl/sycl.hpp>

#include "RightHandSide.hpp"


RightHandSide::RightHandSide(const std::size_t N, const int blockSize, sycl::queue& queue):
    N(N),
    blockSize(blockSize),
    blockCountX(std::ceil(static_cast<double>(N) / static_cast<double>(blockSize))),
    cpuQueue(queue)

{
    // allocate memory for right-hand side storage
    rightHandSideData = sycl::malloc_host<conf::fp_type>(blockCountX * blockSize, queue);
}

RightHandSide::~RightHandSide()
{
    sycl::free(rightHandSideData, cpuQueue);
}
