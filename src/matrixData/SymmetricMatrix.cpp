#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>

#include "SymmetricMatrix.hpp"
#include "Configuration.hpp"

SymmetricMatrix::SymmetricMatrix(const std::size_t N, const std::size_t blockSize): N(N), blockSize(blockSize)
{
}

int SymmetricMatrix::example() {
  conf::fp_type value = 1.0f;
  std::cout << typeid(value).name() << std::endl;
  return 123;
}
