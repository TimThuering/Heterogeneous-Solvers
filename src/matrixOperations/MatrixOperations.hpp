#ifndef MATRIXOPERATIONS_HPP
#define MATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>
#include <vector>

#include "Configuration.hpp"
#include "SymmetricMatrix.hpp"


class MatrixOperations
{
public:
    static void matrixVectorBlock(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result,
                                  int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY);

    static void scaleVectorBlock(sycl::queue& queue, const conf::fp_type* vector, conf::fp_type alpha, conf::fp_type* result,
                                  int blockStart_i, int blockCount_i);
};


#endif //MATRIXOPERATIONS_HPP
