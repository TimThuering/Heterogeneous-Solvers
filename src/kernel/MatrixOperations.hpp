#ifndef MATRIXOPERATIONS_HPP
#define MATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"


class MatrixOperations {
public:
    static sycl::event cholesky(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow);

    static sycl::event cholesky_GPU(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow);

    static sycl::event cholesky_GPU_optimized(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow);
};


#endif //MATRIXOPERATIONS_HPP
