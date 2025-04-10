#ifndef MATRIXOPERATIONS_HPP
#define MATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>
#include <vector>

#include "Configuration.hpp"
#include "SymmetricMatrix.hpp"


class MatrixOperations
{
public:
    static void matrixVectorBlock(sycl::queue& queue, conf::fp_type* A, conf::fp_type* b,
                                  int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j);
};


#endif //MATRIXOPERATIONS_HPP
