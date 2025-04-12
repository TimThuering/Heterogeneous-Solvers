#include "CG.hpp"

#include <iostream>
#include <sycl/sycl.hpp>

#include "MatrixParser.hpp"
#include "MatrixVectorOperations.hpp"

using namespace sycl;

CG::CG(std::string &path_A, std::string &path_b, queue &cpuQueue, queue &gpuQueue) : A(
        MatrixParser::parseSymmetricMatrix(path_A, cpuQueue)),
                                                                                     b(MatrixParser::parseRightHandSide(
                                                                                             path_b, cpuQueue)),
                                                                                     cpuQueue(cpuQueue),
                                                                                     gpuQueue(gpuQueue) {
    // check if dimensions match
    if (A.N != b.N) {
        throw std::invalid_argument(
                "Dimensions of A and b do not match: " + std::to_string(A.N) + " != " + std::to_string(b.N));
    }
    if (A.blockCountXY != b.blockCountX) {
        throw std::invalid_argument(
                "Block count of A and b do not match: " + std::to_string(A.blockCountXY) + " != " +
                std::to_string(b.blockCountX));
    }
}

void CG::solve() {
    const usm_allocator<conf::fp_type, usm::alloc::host> allocator{cpuQueue};
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result(allocator);
    result.resize(b.rightHandSideData.size());


    MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), b.rightHandSideData.data(), result.data(),
                                              0, 0, A.blockCountXY, A.blockCountXY, A.blockCountXY);

    cpuQueue.wait();
    std::string path2 = "../matrixGenerator/matrixBlocked.txt";
    MatrixParser::writeBlockedMatrix(path2, A);
}
