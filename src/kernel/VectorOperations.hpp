#ifndef VECTOROPERATIONS_HPP
#define VECTOROPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"


class VectorOperations
{
public:
    static void scaleVectorBlock(sycl::queue& queue, const conf::fp_type* x, conf::fp_type alpha, conf::fp_type* result,
                                 int blockStart_i, int blockCount_i);

    static void addVectorBlock(sycl::queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                               conf::fp_type* result,
                               int blockStart_i, int blockCount_i);

    static void subVectorBlock(sycl::queue& queue, const conf::fp_type* x, const conf::fp_type* y,
                               conf::fp_type* result,
                               int blockStart_i, int blockCount_i);
};


#endif //VECTOROPERATIONS_HPP
