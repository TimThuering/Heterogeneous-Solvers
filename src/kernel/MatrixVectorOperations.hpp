#ifndef MATRIXOPERATIONS_HPP
#define MATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"


class MatrixVectorOperations
{
public:
    static void matrixVectorBlock(sycl::queue& queue, const conf::fp_type* A, const conf::fp_type* b, conf::fp_type* result,
                                  int blockStart_i, int blockStart_j, int blockCount_i, int blockCount_j, int blockCountXY);
    
};


#endif //MATRIXOPERATIONS_HPP
