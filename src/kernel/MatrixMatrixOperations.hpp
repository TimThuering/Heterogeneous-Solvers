#ifndef MATRIXMATRIXOPERATIONS_HPP
#define MATRIXMATRIXOPERATIONS_HPP

#include <sycl/sycl.hpp>

#include "Configuration.hpp"

class MatrixMatrixOperations {
public:
    static sycl::event triangularSolve(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow, int blockCount);


};



#endif //MATRIXMATRIXOPERATIONS_HPP
