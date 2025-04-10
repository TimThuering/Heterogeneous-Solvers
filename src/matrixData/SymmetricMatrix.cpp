#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>

#include "SymmetricMatrix.hpp"

#include "Configuration.hpp"

SymmetricMatrix::SymmetricMatrix(const std::size_t N, const int blockSize, sycl::queue &queue):
    N(N),
    blockSize(blockSize),
    blockCountXY(std::ceil(static_cast<double>(N) / static_cast<double>(blockSize))),
    cpuQueue(queue),
    matrixData(sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>(queue))
{
    // allocate memory for matrix storage
    const int blockCount = (blockCountXY * (blockCountXY + 1)) / 2.0;
    matrixData.resize(blockCount * blockSize * blockSize);
}
