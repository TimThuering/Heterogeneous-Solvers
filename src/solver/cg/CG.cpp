#include "CG.hpp"

#include <iostream>
#include <sycl/sycl.hpp>

#include "MatrixParser.hpp"
#include "MatrixVectorOperations.hpp"
#include "VectorOperations.hpp"

using namespace sycl;

CG::CG(std::string& path_A, std::string& path_b, queue& cpuQueue, queue& gpuQueue) : A(
        MatrixParser::parseSymmetricMatrix(path_A, cpuQueue)),
    b(MatrixParser::parseRightHandSide(path_b, cpuQueue)),
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

void CG::solve_CPU() {
    const usm_allocator<conf::fp_type, usm::alloc::host> allocatorHost{cpuQueue};

    // result vector x
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> x(allocatorHost);
    x.resize(b.rightHandSideData.size());

    // residual vector r
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> r(allocatorHost);
    r.resize(b.rightHandSideData.size());

    // vector d
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> d(allocatorHost);
    d.resize(b.rightHandSideData.size());

    // vector q
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> q(allocatorHost);
    q.resize(b.rightHandSideData.size());

    // temporary vector
    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> tmp(allocatorHost);
    tmp.resize(b.rightHandSideData.size());

    const int workGroupCountScalarProduct = (b.blockCountX * conf::matrixBlockSize / 2) / conf::workGroupSizeVector;

    conf::fp_type delta_new = 0;
    conf::fp_type delta_old = 0;
    conf::fp_type delta_zero = 0;
    conf::fp_type alpha = 0;
    conf::fp_type beta = 0;

    conf::fp_type epsilon2 = conf::epsilon * conf::epsilon;


    // r = b - Ax
    MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), x.data(), r.data(),
                                              0, 0, A.blockCountXY, A.blockCountXY, A.blockCountXY);
    cpuQueue.wait();
    VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r.data(), r.data(), 0, A.blockCountXY);
    cpuQueue.wait();

    // d = r
    std::memcpy(d.data(), r.data(), sizeof(conf::fp_type) * b.rightHandSideData.size());

    // δ_new = r^T * r
    // δ_0 = δ_new
    VectorOperations::scalarProduct(cpuQueue, r.data(), r.data(), tmp.data(), 0, A.blockCountXY);
    cpuQueue.wait();
    VectorOperations::sumFinalScalarProduct(cpuQueue, tmp.data(), workGroupCountScalarProduct);
    cpuQueue.wait();
    delta_new = tmp[0];
    delta_zero = delta_new;

    std::size_t iteration = 0;

    while (iteration < conf::iMax && delta_new > epsilon2 * delta_zero) {
        // q = Ad
        MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), d.data(), q.data(), 0, 0,
                                                  A.blockCountXY, A.blockCountXY, A.blockCountXY);
        cpuQueue.wait();

        // 𝛼 = δ_new / d^T * q
        VectorOperations::scalarProduct(cpuQueue, d.data(), q.data(), tmp.data(), 0, A.blockCountXY);
        cpuQueue.wait();
        VectorOperations::sumFinalScalarProduct(cpuQueue, tmp.data(), workGroupCountScalarProduct);
        cpuQueue.wait();
        alpha = delta_new / tmp[0];

        // x = x + 𝛼d
        VectorOperations::scaleVectorBlock(cpuQueue, d.data(), alpha, tmp.data(), 0, b.blockCountX);
        cpuQueue.wait();
        VectorOperations::addVectorBlock(cpuQueue, x.data(), tmp.data(), x.data(), 0, b.blockCountX);
        cpuQueue.wait();

        if (iteration % 50 == 0) {
            // compute real residual every 50 iterations --> requires additional matrix vector product

            // r = b - Ax
            MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), x.data(), r.data(),
                                                      0, 0, A.blockCountXY, A.blockCountXY, A.blockCountXY);
            cpuQueue.wait();
            VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r.data(), r.data(), 0, A.blockCountXY);
            cpuQueue.wait();
        } else {
            // compute residual without an additional matrix vector product

            // r = r - 𝛼q
            VectorOperations::scaleVectorBlock(cpuQueue, q.data(), alpha, tmp.data(), 0, b.blockCountX);
            cpuQueue.wait();
            VectorOperations::subVectorBlock(cpuQueue, r.data(), tmp.data(), r.data(), 0, b.blockCountX);
            cpuQueue.wait();
        }

        // δ_old = δ_new
        delta_old = delta_new;

        // δ_new = r^T * r
        VectorOperations::scalarProduct(cpuQueue, r.data(), r.data(), tmp.data(), 0, A.blockCountXY);
        cpuQueue.wait();
        VectorOperations::sumFinalScalarProduct(cpuQueue, tmp.data(), workGroupCountScalarProduct);
        cpuQueue.wait();
        delta_new = tmp[0];

        // β = δ_new / δ_old
        beta = delta_new / delta_old;

        // d = r + βd
        VectorOperations::scaleVectorBlock(cpuQueue, d.data(), beta, d.data(), 0, b.blockCountX);
        cpuQueue.wait();
        VectorOperations::addVectorBlock(cpuQueue, r.data(), d.data(), d.data(), 0, b.blockCountX);
        cpuQueue.wait();

        iteration++;

        // std::cout << "iteration: " << iteration << std::endl;
        std::cout << delta_new << std::endl;
    }


    cpuQueue.wait();
}
