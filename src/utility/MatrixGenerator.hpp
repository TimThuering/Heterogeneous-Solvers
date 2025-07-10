#ifndef MATRIXGENERATOR_HPP
#define MATRIXGENERATOR_HPP

#include <sycl/sycl.hpp>

#include "RightHandSide.hpp"
#include "SymmetricMatrix.hpp"


class MatrixGenerator {
public:
    static SymmetricMatrix generateSPDMatrixStrictDiagonalDominant(sycl::queue& queue);
    static SymmetricMatrix generateSPDMatrix(std::string& path, sycl::queue& queue);
    static void generateTestKernelMatrix(std::string& path_train, std::string& path_test, sycl::queue& queue, conf::fp_type* K_star);
    static RightHandSide parseRHS_GP(std::string& path, sycl::queue& queue);
    static RightHandSide generateRHS(sycl::queue& queue);

private:
    static void readInputVector(std::string& path,std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>>& dataVector, int N, int offset);
};


#endif //MATRIXGENERATOR_HPP
