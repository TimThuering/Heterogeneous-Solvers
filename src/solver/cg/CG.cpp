#include "CG.hpp"

#include <iostream>
#include <sycl/sycl.hpp>

#include "MatrixParser.hpp"
#include "MatrixOperations.hpp"
using namespace sycl;

CG::CG(std::string& path_A, std::string& path_b, queue cpuQueue, queue gpuQueue): A(MatrixParser::parseSymmetricMatrix(path_A, cpuQueue)),
                                                  b(MatrixParser::parseRightHandSide(path_b, cpuQueue))
{
    // check if dimensions match
    if (A.N != b.N)
    {
        throw std::invalid_argument(
            "Dimensions of A and b do not match: " + std::to_string(A.N) + " != " + std::to_string(b.N));
    }
    if(A.blockCountXY != b.blockCountX)
    {
        throw std::invalid_argument(
           "Block count of A and b do not match: " + std::to_string(A.blockCountXY) + " != " + std::to_string(b.blockCountX));
    }
}

void CG::solve()
{
    std::cout << b.N << std::endl;
    std::cout << b.rightHandSideData[0] << std::endl;


    // conf::fp_type *testData = malloc_host<conf::fp_type>(A.matrixData.size(), cpu_queue);

    // MatrixOperations::matrixVectorBlock(cpuQueue, A.matrixData, b.rightHandSideData, 0,0,A.blockCountXY,A.blockSize);
}
