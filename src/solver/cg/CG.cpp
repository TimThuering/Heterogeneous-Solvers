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
    x(sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>(cpuQueue)),
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

    // resize result vector x to correct size
    x.resize(b.rightHandSideData.size());
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


    unsigned int workGroupCountScalarProduct;

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
    workGroupCountScalarProduct = VectorOperations::scalarProduct(cpuQueue, r.data(), r.data(), tmp.data(), 0,
                                                                  A.blockCountXY);
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
        workGroupCountScalarProduct = VectorOperations::scalarProduct(cpuQueue, d.data(), q.data(), tmp.data(), 0,
                                                                      A.blockCountXY);
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
        workGroupCountScalarProduct = VectorOperations::scalarProduct(cpuQueue, r.data(), r.data(), tmp.data(), 0,
                                                                      A.blockCountXY);
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

        std::cout << (iteration - 1) << ": Iteration time: " << iterationTime << "ms" << std::endl;
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


    unsigned int workGroupCountScalarProduct;

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
    workGroupCountScalarProduct = VectorOperations::scalarProduct(gpuQueue, r, r, tmp, 0, A.blockCountXY);
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
        workGroupCountScalarProduct = VectorOperations::scalarProduct(gpuQueue, d, q, tmp, 0, A.blockCountXY);
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
        workGroupCountScalarProduct = VectorOperations::scalarProduct(gpuQueue, r, r, tmp, 0, A.blockCountXY);
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

        std::cout << (iteration - 1) << ": Iteration time: " << iterationTime << "ms" << std::endl;
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

void CG::solveHeterogeneous_static() {
    const auto start = std::chrono::steady_clock::now();

    // static split:
    constexpr double gpuProportion = 0.5;
    blockCountGPU = std::ceil(static_cast<double>(A.blockCountXY) * gpuProportion);
    blockCountCPU = A.blockCountXY - blockCountGPU;
    blockStartCPU = blockCountGPU;

    // Total amount of blocks needed for the upper part of the matrix
    const std::size_t blockCountGPUTotal = blockCountGPU * A.blockCountXY;

    // initialize data structures
    initGPUdataStructures(blockCountGPU, blockCountGPUTotal);
    initCPUdataStructures();

    // variables for cg algorithm
    conf::fp_type delta_new = 0;
    conf::fp_type delta_old = 0;
    conf::fp_type delta_zero = 0;
    conf::fp_type alpha = 0;
    conf::fp_type beta = 0;

    conf::fp_type epsilon2 = conf::epsilon * conf::epsilon;

    // initial calculations of the CG algorithm:
    //      r = b - Ax
    //      d = r
    //      δ_new = r^T * r
    //      δ_0 = δ_new
    initCG(delta_zero, delta_new);

    std::size_t iteration = 0;

    while (iteration < conf::iMax && delta_new > epsilon2 * delta_zero) {
        auto startIteration = std::chrono::steady_clock::now();

        // q = Ad
        compute_q();

        // 𝛼 = δ_new / d^T * q
        compute_alpha(alpha, delta_new);

        // x = x + 𝛼d
        update_x(alpha);

        if (iteration % 50 == 0) {
            // compute real residual every 50 iterations --> requires additional matrix vector product

            // r = b - Ax
            computeRealResidual();
        } else {
            // compute residual without an additional matrix vector product

            // r = r - 𝛼q
            update_r(alpha);
        }
        // δ_old = δ_new
        delta_old = delta_new;

        // δ_new = r^T * r
        compute_delta_new(delta_new);

        // β = δ_new / δ_old
        beta = delta_new / delta_old;

        // d = r + βd
        compute_d(beta);

        iteration++;

        auto endIteration = std::chrono::steady_clock::now();
        auto iterationTime = std::chrono::duration<double, std::milli>(endIteration - startIteration).count();

        std::cout << (iteration - 1) << ": Iteration time: " << iterationTime << "ms" << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << totalTime << "ms (" << iteration << " iterations)" << std::endl;
    std::cout << "Residual: " << delta_new << std::endl;

    gpuQueue.wait();
    cpuQueue.wait();

    gpuQueue.submit([&](handler& h) {
        h.memcpy(x.data(), x_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    }).wait();
}

void CG::initGPUdataStructures(const std::size_t blockCountGPU, const std::size_t blockCountGPUTotal) {
    // Matrix A GPU
    A_gpu = malloc_device<conf::fp_type>(blockCountGPUTotal * conf::matrixBlockSize * conf::matrixBlockSize,
                                         gpuQueue);
    gpuQueue.submit([&](handler& h) {
        h.memcpy(A_gpu, A.matrixData.data(),
                 blockCountGPUTotal * conf::matrixBlockSize * conf::matrixBlockSize * sizeof(conf::fp_type));
    }).wait();

    // Right-hand side b GPU
    b_gpu = malloc_device<conf::fp_type>(blockCountGPU * conf::matrixBlockSize, gpuQueue);
    gpuQueue.submit([&](handler& h) {
        h.memcpy(b_gpu, b.rightHandSideData.data(), blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    }).wait();

    // result vector x
    x_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // residual vector r
    r_gpu = malloc_device<conf::fp_type>(blockCountGPU * conf::matrixBlockSize, gpuQueue);

    // vector d
    d_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // vector q
    q_gpu = malloc_device<conf::fp_type>(blockCountGPU * conf::matrixBlockSize, gpuQueue);

    // temporary vector
    tmp_gpu = malloc_device<conf::fp_type>(blockCountGPU * conf::matrixBlockSize, gpuQueue);
}

void CG::initCPUdataStructures() {
    const usm_allocator<conf::fp_type, usm::alloc::host> allocatorHost{cpuQueue};


    // residual vector r
    r_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), cpuQueue);

    // vector d
    d_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), cpuQueue);

    // vector q
    q_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), cpuQueue);

    // temporary vector
    tmp_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), cpuQueue);
}

void CG::freeDataStructures() {
    sycl::free(A_gpu, gpuQueue);
    sycl::free(b_gpu, gpuQueue);
    sycl::free(x_gpu, gpuQueue);
    sycl::free(r_gpu, gpuQueue);
    sycl::free(d_gpu, gpuQueue);
    sycl::free(q_gpu, gpuQueue);
    sycl::free(tmp_gpu, gpuQueue);

    sycl::free(r_cpu, cpuQueue);
    sycl::free(d_cpu, cpuQueue);
    sycl::free(q_cpu, cpuQueue);
    sycl::free(tmp_cpu, cpuQueue);
}

void CG::initCG(conf::fp_type& delta_zero, conf::fp_type& delta_new) {
    // r = b - Ax
    MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, x_gpu, r_gpu,
                                              0, 0, blockCountGPU, A.blockCountXY, A.blockCountXY);
    MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), x.data(), r_cpu,
                                              blockStartCPU, 0, blockCountCPU, A.blockCountXY, A.blockCountXY);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::subVectorBlock(gpuQueue, b_gpu, r_gpu, r_gpu, 0, blockCountGPU);
    VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r_cpu, r_cpu, blockStartCPU,
                                     blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();

    // d = r
    gpuQueue.submit([&](handler& h) {
        h.memcpy(d_gpu, r_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    });
    cpuQueue.submit([&](handler& h) {
        h.memcpy(&d_cpu[blockStartCPU * conf::matrixBlockSize], &r_cpu[blockStartCPU * conf::matrixBlockSize],
                 blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    });
    gpuQueue.wait();
    cpuQueue.wait();

    // δ_new = r^T * r
    // δ_0 = δ_new
    const unsigned int workGroupCountScalarProduct_GPU =
        VectorOperations::scalarProduct(gpuQueue, r_gpu, r_gpu, tmp_gpu, 0, blockCountGPU);
    const unsigned int workGroupCountScalarProduct_CPU =
        VectorOperations::scalarProduct(cpuQueue, r_cpu, r_cpu, tmp_cpu, blockStartCPU, blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    VectorOperations::sumFinalScalarProduct(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    gpuQueue.wait();
    cpuQueue.wait();

    // get value of δ_new from gpu
    gpuQueue.submit([&](handler& h) { h.memcpy(&delta_new, tmp_gpu, sizeof(conf::fp_type)); }).wait();
    delta_new = delta_new + tmp_cpu[0];
    delta_zero = delta_new;
}

void CG::compute_q() {
    // exchange parts of d vector so that both CPU and GPU hold the complete vector
    gpuQueue.submit([&](handler& h) {
        h.memcpy(&d_gpu[blockStartCPU * conf::matrixBlockSize], &d_cpu[blockStartCPU * conf::matrixBlockSize],
                 blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    });
    gpuQueue.submit([&](handler& h) {
        h.memcpy(d_cpu, d_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    });
    gpuQueue.wait();
    cpuQueue.wait();

    // q = Ad
    MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, d_gpu, q_gpu, 0, 0,
                                              blockCountGPU, A.blockCountXY, A.blockCountXY);
    MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), d_cpu, q_cpu,
                                              blockStartCPU, 0,
                                              blockCountCPU, A.blockCountXY, A.blockCountXY);
    gpuQueue.wait();
    cpuQueue.wait();
}

void CG::compute_alpha(conf::fp_type& alpha, conf::fp_type& delta_new) {
    // 𝛼 = δ_new / d^T * q
    const unsigned int workGroupCountScalarProduct_GPU = VectorOperations::scalarProduct(
        gpuQueue, d_gpu, q_gpu, tmp_gpu, 0, blockCountGPU);
    const unsigned int workGroupCountScalarProduct_CPU = VectorOperations::scalarProduct(
        cpuQueue, d_cpu, q_cpu, tmp_cpu, blockStartCPU,
        blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    VectorOperations::sumFinalScalarProduct(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    gpuQueue.wait();
    cpuQueue.wait();
    conf::fp_type result = 0;
    gpuQueue.submit([&](handler& h) {
        h.memcpy(&result, tmp_gpu, sizeof(conf::fp_type));
    }).wait();
    result = result + tmp_cpu[0];
    alpha = delta_new / result;
}

void CG::update_x(conf::fp_type alpha) {
    // x = x + 𝛼d
    VectorOperations::scaleVectorBlock(gpuQueue, d_gpu, alpha, tmp_gpu, 0, blockCountGPU);
    VectorOperations::scaleVectorBlock(cpuQueue, d_cpu, alpha, tmp_cpu, blockStartCPU, blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::addVectorBlock(gpuQueue, x_gpu, tmp_gpu, x_gpu, 0, blockCountGPU);
    VectorOperations::addVectorBlock(cpuQueue, x.data(), tmp_cpu, x.data(), blockStartCPU,
                                     blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
}

void CG::computeRealResidual() {
    // exchange parts of x vector so that both CPU and GPU hold the complete vector
    gpuQueue.submit([&](handler& h) {
        h.memcpy(&x_gpu[blockStartCPU * conf::matrixBlockSize], &x[blockStartCPU * conf::matrixBlockSize],
                 blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    });
    gpuQueue.submit([&](handler& h) {
        h.memcpy(x.data(), x_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    });
    gpuQueue.wait();
    cpuQueue.wait();


    // r = b - Ax
    MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, x_gpu, r_gpu,
                                              0, 0, blockCountGPU, A.blockCountXY, A.blockCountXY);
    MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), x.data(), r_cpu,
                                              blockStartCPU, 0, blockCountCPU, A.blockCountXY, A.blockCountXY);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::subVectorBlock(gpuQueue, b_gpu, r_gpu, r_gpu, 0, blockCountGPU);
    VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r_cpu, r_cpu,
                                     blockStartCPU, blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
}

void CG::update_r(conf::fp_type alpha) {
    // r = r - 𝛼q
    VectorOperations::scaleVectorBlock(gpuQueue, q_gpu, alpha, tmp_gpu, 0, blockCountGPU);
    VectorOperations::scaleVectorBlock(cpuQueue, q_cpu, alpha, tmp_cpu, blockStartCPU,
                                       blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::subVectorBlock(gpuQueue, r_gpu, tmp_gpu, r_gpu, 0, blockCountGPU);
    VectorOperations::subVectorBlock(cpuQueue, r_cpu, tmp_cpu, r_cpu, blockStartCPU,
                                     blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
}

void CG::compute_delta_new(conf::fp_type& delta_new) {
    // δ_new = r^T * r
    const unsigned int workGroupCountScalarProduct_GPU = VectorOperations::scalarProduct(
        gpuQueue, r_gpu, r_gpu, tmp_gpu, 0, blockCountGPU);
    const unsigned int workGroupCountScalarProduct_CPU = VectorOperations::scalarProduct(
        cpuQueue, r_cpu, r_cpu, tmp_cpu, blockStartCPU,
        blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    VectorOperations::sumFinalScalarProduct(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    gpuQueue.wait();
    cpuQueue.wait();
    // get value of δ_new from gpu
    gpuQueue.submit([&](handler& h) {
        h.memcpy(&delta_new, tmp_gpu, sizeof(conf::fp_type));
    }).wait();
    delta_new = delta_new + tmp_cpu[0];
}

void CG::compute_d(conf::fp_type& beta) {
    // d = r + βd
    VectorOperations::scaleVectorBlock(gpuQueue, d_gpu, beta, d_gpu, 0, blockCountGPU);
    VectorOperations::scaleVectorBlock(cpuQueue, d_cpu, beta, d_cpu, blockStartCPU, blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
    VectorOperations::addVectorBlock(gpuQueue, r_gpu, d_gpu, d_gpu, 0, blockCountGPU);
    VectorOperations::addVectorBlock(cpuQueue, r_cpu, d_cpu, d_cpu, blockStartCPU,
                                     blockCountCPU);
    gpuQueue.wait();
    cpuQueue.wait();
}
