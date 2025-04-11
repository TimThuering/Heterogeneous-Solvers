#ifndef MATRIXOPERATIONS_HPP
#define MATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>
#include <vector>

#include "Configuration.hpp"
#include "SymmetricMatrix.hpp"


class MatrixVectorOperations
{
public:
    static void matrixVectorBlock(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result,
                                  int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY);

    static void scaleVectorBlock(sycl::queue& queue, const conf::fp_type* x, conf::fp_type alpha, conf::fp_type* result,
                                  int blockStart_i, int blockCount_i);

    static void addVectorBlock(sycl::queue& queue, const conf::fp_type* x, const conf::fp_type* y, conf::fp_type* result,
                              int blockStart_i, int blockCount_i);

    static void subVectorBlock(sycl::queue& queue, const conf::fp_type* x, const conf::fp_type* y, conf::fp_type* result,
                          int blockStart_i, int blockCount_i);
};


#endif //MATRIXOPERATIONS_HPP
