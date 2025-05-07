#ifndef MATRIXGENERATOR_HPP
#define MATRIXGENERATOR_HPP

#include <sycl/sycl.hpp>

#include "RightHandSide.hpp"
#include "SymmetricMatrix.hpp"


class MatrixGenerator {
public:
    static SymmetricMatrix generateSPDMatrix(sycl::queue& queue);
    static RightHandSide generateRHS(sycl::queue& queue);

};



#endif //MATRIXGENERATOR_HPP
