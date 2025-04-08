#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>

#include "SymmetricMatrix.hpp"
#include "Configuration.hpp"

SymmetricMatrix::SymmetricMatrix(const std::size_t N, const int blockSize):
    N(N),
    blockSize(blockSize),
    blockCountXY(std::ceil(static_cast<double>(N) / static_cast<double>(blockSize)))
{
    // allocate memory for matrix storage
    matrixData.resize(blockCountXY * blockCountXY * blockSize * blockSize);
}
