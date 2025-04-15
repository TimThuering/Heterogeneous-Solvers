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
    auto start = std::chrono::steady_clock::now();

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

    if ((b.blockCountX * conf::matrixBlockSize / 2) % conf::workGroupSizeVector != 0) {
        throw std::invalid_argument("Wrong work group size and block size combination");
    }
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
        auto startIteration = std::chrono::steady_clock::now();


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
            VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r.data(), r.data(), 0,
                                             A.blockCountXY);
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

        auto endIteration = std::chrono::steady_clock::now();
        auto iterationTime = std::chrono::duration<double, std::milli>(endIteration - startIteration).count();

        std::cout << "Iteration time: " << iterationTime << "ms" << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << totalTime << "ms (" << iteration << " iterations)" << std::endl;

    cpuQueue.wait();
}

void CG::solve_GPU() {
    auto start = std::chrono::steady_clock::now();

    // Matrix A GPU
    auto* A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);
    gpuQueue.submit([&](handler& h) {
        h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
    }).wait();

    // Right-hand side b GPU
    auto* b_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);
    gpuQueue.submit([&](handler& h) {
        h.memcpy(b_gpu, b.rightHandSideData.data(), b.rightHandSideData.size() * sizeof(conf::fp_type));
    }).wait();

    // result vector x
    auto* x = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // residual vector r
    auto* r = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // vector d
    auto* d = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // vector q
    auto* q = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // temporary vector
    auto* tmp = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);


    if ((b.blockCountX * conf::matrixBlockSize / 2) % conf::workGroupSizeVector != 0) {
        throw std::invalid_argument("Wrong work group size and block size combination");
    }
    const int workGroupCountScalarProduct = (b.blockCountX * conf::matrixBlockSize / 2) / conf::workGroupSizeVector;

    conf::fp_type delta_new = 0;
    conf::fp_type delta_old = 0;
    conf::fp_type delta_zero = 0;
    conf::fp_type alpha = 0;
    conf::fp_type beta = 0;

    conf::fp_type epsilon2 = conf::epsilon * conf::epsilon;


    // r = b - Ax
    MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, x, r,
                                              0, 0, A.blockCountXY, A.blockCountXY, A.blockCountXY);
    gpuQueue.wait();
    VectorOperations::subVectorBlock(gpuQueue, b_gpu, r, r, 0, A.blockCountXY);
    gpuQueue.wait();

    // d = r
    gpuQueue.submit([&](handler& h) {
        h.memcpy(d, r, b.rightHandSideData.size() * sizeof(conf::fp_type));
    }).wait();

    // δ_new = r^T * r
    // δ_0 = δ_new
    VectorOperations::scalarProduct(gpuQueue, r, r, tmp, 0, A.blockCountXY);
    gpuQueue.wait();
    VectorOperations::sumFinalScalarProduct(gpuQueue, tmp, workGroupCountScalarProduct);
    gpuQueue.wait();

    // get value of δ_new from gpu
    gpuQueue.submit([&](handler& h) {
        h.memcpy(&delta_new, tmp, sizeof(conf::fp_type));
    }).wait();
    delta_zero = delta_new;

    std::size_t iteration = 0;

    while (iteration < conf::iMax && delta_new > epsilon2 * delta_zero) {
        auto startIteration = std::chrono::steady_clock::now();


        // q = Ad
        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, d, q, 0, 0,
                                                  A.blockCountXY, A.blockCountXY, A.blockCountXY);
        gpuQueue.wait();

        // 𝛼 = δ_new / d^T * q
        VectorOperations::scalarProduct(gpuQueue, d, q, tmp, 0, A.blockCountXY);
        gpuQueue.wait();
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp, workGroupCountScalarProduct);
        gpuQueue.wait();
        conf::fp_type result = 0;
        gpuQueue.submit([&](handler& h) {
            h.memcpy(&result, tmp, sizeof(conf::fp_type));
        }).wait();
        alpha = delta_new / result;

        // x = x + 𝛼d
        VectorOperations::scaleVectorBlock(gpuQueue, d, alpha, tmp, 0, b.blockCountX);
        gpuQueue.wait();
        VectorOperations::addVectorBlock(gpuQueue, x, tmp, x, 0, b.blockCountX);
        gpuQueue.wait();

        if (iteration % 50 == 0) {
            // compute real residual every 50 iterations --> requires additional matrix vector product

            // r = b - Ax
            MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, x, r,
                                                      0, 0, A.blockCountXY, A.blockCountXY, A.blockCountXY);
            gpuQueue.wait();
            VectorOperations::subVectorBlock(gpuQueue, b_gpu, r, r, 0,
                                             A.blockCountXY);
            gpuQueue.wait();
        } else {
            // compute residual without an additional matrix vector product

            // r = r - 𝛼q
            VectorOperations::scaleVectorBlock(gpuQueue, q, alpha, tmp, 0, b.blockCountX);
            gpuQueue.wait();
            VectorOperations::subVectorBlock(gpuQueue, r, tmp, r, 0, b.blockCountX);
            gpuQueue.wait();
        }

        // δ_old = δ_new
        delta_old = delta_new;

        // δ_new = r^T * r
        VectorOperations::scalarProduct(gpuQueue, r, r, tmp, 0, A.blockCountXY);
        gpuQueue.wait();
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp, workGroupCountScalarProduct);
        gpuQueue.wait();
        // get value of δ_new from gpu
        gpuQueue.submit([&](handler& h) {
            h.memcpy(&delta_new, tmp, sizeof(conf::fp_type));
        }).wait();
        // β = δ_new / δ_old
        beta = delta_new / delta_old;

        // d = r + βd
        VectorOperations::scaleVectorBlock(gpuQueue, d, beta, d, 0, b.blockCountX);
        gpuQueue.wait();
        VectorOperations::addVectorBlock(gpuQueue, r, d, d, 0, b.blockCountX);
        gpuQueue.wait();

        iteration++;

        auto endIteration = std::chrono::steady_clock::now();
        auto iterationTime = std::chrono::duration<double, std::milli>(endIteration - startIteration).count();

        std::cout << "Iteration time: " << iterationTime << "ms" << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << totalTime << "ms (" << iteration << " iterations)" << std::endl;

    gpuQueue.wait();

    std::vector<conf::fp_type> result(b.rightHandSideData.size());
    gpuQueue.submit([&](handler& h) {
        h.memcpy(result.data(), x, b.rightHandSideData.size() * sizeof(conf::fp_type));
    }).wait();

    sycl::free(A_gpu, gpuQueue);
    sycl::free(b_gpu, gpuQueue);
    sycl::free(x, gpuQueue);
    sycl::free(r, gpuQueue);
    sycl::free(d, gpuQueue);
    sycl::free(q, gpuQueue);
    sycl::free(tmp, gpuQueue);
}
