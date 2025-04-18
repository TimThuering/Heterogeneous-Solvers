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
    x(sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>(cpuQueue)),
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

void CG::solveHeterogeneous_static() {
    const auto start = std::chrono::steady_clock::now();

    // static split:
    constexpr double gpuProportion = 0;
    blockCountGPU = std::ceil(static_cast<double>(A.blockCountXY) * gpuProportion);
    blockCountCPU = A.blockCountXY - blockCountGPU;
    blockStartCPU = blockCountGPU;

    std::cout << "Block count GPU: " << blockCountGPU << std::endl;
    std::cout << "Block count CPU: " << blockCountCPU << std::endl;

    // Total amount of blocks needed for the upper part of the matrix
    const std::size_t blockCountGPUTotal = (A.blockCountXY * (A.blockCountXY + 1) / 2) - (blockCountCPU * (blockCountCPU
        + 1) / 2);

    // initialize data structures
    initGPUdataStructures(blockCountGPUTotal);
    initCPUdataStructures();

    // variables for cg algorithm
    conf::fp_type delta_new = 0;
    conf::fp_type delta_old = 0;
    conf::fp_type delta_zero = 0;
    conf::fp_type alpha = 0;
    conf::fp_type beta = 0;

    conf::fp_type epsilon2 = conf::epsilon * conf::epsilon;

    /*
     * initial calculations of the CG algorithm:
     *     r = b - Ax
     *     d = r
     *     δ_new = r^T * r
     *     δ_0 = δ_new
     */
    initCG(delta_zero, delta_new);

    std::size_t iteration = 0;

    while (iteration < conf::iMax && delta_new > epsilon2 * delta_zero) {
        auto startIteration = std::chrono::steady_clock::now();

        // compute_q(); // q = Ad
        compute_q(); // q = Ad
        compute_alpha(alpha, delta_new); // 𝛼 = δ_new / d^T * q
        update_x(alpha); // x = x + 𝛼d

        if (iteration % 50 == 0) {
            // compute real residual every 50 iterations --> requires additional matrix vector product
            computeRealResidual(); // r = b - Ax
        } else {
            // compute residual without an additional matrix vector product
            update_r(alpha); // r = r - 𝛼q
        }
        delta_old = delta_new; // δ_old = δ_new
        compute_delta_new(delta_new); // δ_new = r^T * r
        beta = delta_new / delta_old; // β = δ_new / δ_old
        compute_d(beta); // d = r + βd

        iteration++;

        auto endIteration = std::chrono::steady_clock::now();
        auto iterationTime = std::chrono::duration<double, std::milli>(endIteration - startIteration).count();
        std::cout << (iteration - 1) << ": Iteration time: " << iterationTime << "ms" << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << totalTime << "ms (" << iteration << " iterations)" << std::endl;
    std::cout << "Residual: " << delta_new << std::endl;

    waitAllQueues();

    if (blockCountGPU != 0) {
        gpuQueue.submit([&](handler& h) {
            h.memcpy(x.data(), x_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        }).wait();
    }
}

void CG::initGPUdataStructures(const std::size_t blockCountGPUTotal) {
    if (blockCountGPU == 0) {
        return;
    }

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
    if (blockCountCPU == 0) {
        return;
    }

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
    if (blockCountGPU != 0) {
        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, x_gpu, r_gpu,
                                                  0, 0, blockCountGPU, A.blockCountXY, A.blockCountXY);
    }
    if (blockCountCPU != 0) {
        MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), x.data(), r_cpu, blockStartCPU, 0,
                                                  blockCountCPU, A.blockCountXY, A.blockCountXY);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::subVectorBlock(gpuQueue, b_gpu, r_gpu, r_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r_cpu, r_cpu, blockStartCPU,
                                         blockCountCPU);
    }
    waitAllQueues();

    // d = r
    if (blockCountGPU != 0) {
        gpuQueue.submit([&](handler& h) {
            h.memcpy(d_gpu, r_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
    }
    if (blockCountCPU != 0) {
        cpuQueue.submit([&](handler& h) {
            h.memcpy(&d_cpu[blockStartCPU * conf::matrixBlockSize], &r_cpu[blockStartCPU * conf::matrixBlockSize],
                     blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
    }
    waitAllQueues();

    // δ_new = r^T * r
    // δ_0 = δ_new
    unsigned int workGroupCountScalarProduct_GPU = 0;
    unsigned int workGroupCountScalarProduct_CPU = 0;
    if (blockCountGPU != 0) {
        workGroupCountScalarProduct_GPU =
            VectorOperations::scalarProduct(gpuQueue, r_gpu, r_gpu, tmp_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        workGroupCountScalarProduct_CPU =
            VectorOperations::scalarProduct(cpuQueue, r_cpu, r_cpu, tmp_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::sumFinalScalarProduct(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    }
    waitAllQueues();


    delta_new = 0;
    if (blockCountGPU != 0) {
        // get value of δ_new from gpu
        gpuQueue.submit([&](handler& h) { h.memcpy(&delta_new, tmp_gpu, sizeof(conf::fp_type)); }).wait();
    }
    if (blockCountCPU != 0) {
        delta_new = delta_new + tmp_cpu[0];
    }
    delta_zero = delta_new;
}

void CG::compute_q() {

    // auto startMV = std::chrono::steady_clock::now();

    if (blockCountGPU != 0 && blockCountCPU != 0) {
        // exchange parts of d vector so that both CPU and GPU hold the complete vector
        gpuQueue.submit([&](handler& h) {
            h.memcpy(&d_gpu[blockStartCPU * conf::matrixBlockSize], &d_cpu[blockStartCPU * conf::matrixBlockSize],
                     blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
        gpuQueue.submit([&](handler& h) {
            h.memcpy(d_cpu, d_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
    }

    waitAllQueues();

    // auto endMV = std::chrono::steady_clock::now();
    // auto iterationTime = std::chrono::duration<double, std::milli>(endMV - startMV).count();
    // std::cout << "--------------------------------- MV time: " << iterationTime << "ms" << std::endl;

    // q = Ad
    if (blockCountGPU != 0) {
        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, d_gpu, q_gpu, 0, 0,
                                                  blockCountGPU, A.blockCountXY, A.blockCountXY);
    }
    if (blockCountCPU != 0) {

        MatrixVectorOperations::matrixVectorBlock_CPU(cpuQueue, A.matrixData.data(), d_cpu, q_cpu, blockStartCPU, 0,
                                                  blockCountCPU, A.blockCountXY, A.blockCountXY);

    }
    waitAllQueues();
}

void CG::compute_q_CommunicationHiding() {
    if (blockCountGPU != 0 && blockCountCPU != 0) {
        // auto startMV = std::chrono::steady_clock::now();


        queue memQueue(gpuQueue.get_device());

        // exchange parts of d vector so that both CPU and GPU hold the complete vector. Happens asynchronously.
        memQueue.submit([&](handler& h) {
            h.memcpy(&d_gpu[blockStartCPU * conf::matrixBlockSize], &d_cpu[blockStartCPU * conf::matrixBlockSize],
                     blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
        memQueue.submit([&](handler& h) {
            h.memcpy(d_cpu, d_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });


        // perform partial matrix-vector products with data that is already on the GPU / CPU

        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, d_gpu, q_gpu, 0, 0,
                                                  blockCountGPU, blockCountGPU, A.blockCountXY);
        MatrixVectorOperations::matrixVectorBlock_CPU(cpuQueue, A.matrixData.data(), d_cpu, q_cpu, blockStartCPU,
                                                  blockStartCPU,
                                                  blockCountCPU, blockCountCPU, A.blockCountXY);

        // wait for memory transfers to finish
        waitAllQueues();
        memQueue.wait();


        // perform missing partial matrix-vector products with the transferred data
        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, d_gpu, q_gpu, 0, blockCountGPU,
                                                  blockCountGPU, blockCountCPU, A.blockCountXY, false);
        MatrixVectorOperations::matrixVectorBlock_CPU(cpuQueue, A.matrixData.data(), d_cpu, q_cpu, blockStartCPU, 0,
                                                  blockCountCPU, blockCountGPU, A.blockCountXY, false);
        waitAllQueues();

    } else if (blockCountGPU != 0 && blockCountCPU == 0) {
        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, d_gpu, q_gpu, 0, 0,
                                                  blockCountGPU, A.blockCountXY, A.blockCountXY);
        waitAllQueues();
    } else if (blockCountCPU != 0 && blockCountGPU == 0) {
        MatrixVectorOperations::matrixVectorBlock_CPU(cpuQueue, A.matrixData.data(), d_cpu, q_cpu, blockStartCPU, 0,
                                                  blockCountCPU, A.blockCountXY, A.blockCountXY);
        waitAllQueues();
    } else {
        throw std::runtime_error("Invalid CPU/GPU proportion");
    }
}

void CG::compute_alpha(conf::fp_type& alpha, conf::fp_type& delta_new) {
    unsigned int workGroupCountScalarProduct_GPU = 0;
    unsigned int workGroupCountScalarProduct_CPU = 0;

    // 𝛼 = δ_new / d^T * q
    if (blockCountGPU != 0) {
        workGroupCountScalarProduct_GPU = VectorOperations::scalarProduct(
            gpuQueue, d_gpu, q_gpu, tmp_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        workGroupCountScalarProduct_CPU = VectorOperations::scalarProduct(
            cpuQueue, d_cpu, q_cpu, tmp_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::sumFinalScalarProduct(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    }
    waitAllQueues();


    conf::fp_type result = 0;
    if (blockCountGPU != 0) {
        gpuQueue.submit([&](handler& h) { h.memcpy(&result, tmp_gpu, sizeof(conf::fp_type)); }).wait();
    }
    if (blockCountCPU != 0) {
        result = result + tmp_cpu[0];
    }

    alpha = delta_new / result;
}

void CG::update_x(conf::fp_type alpha) {
    // x = x + 𝛼d
    if (blockCountGPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(gpuQueue, x_gpu, alpha, d_gpu, x_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(cpuQueue, x.data(), alpha, d_cpu, x.data(), blockStartCPU,
                                                 blockCountCPU);
    }
    waitAllQueues();
}

void CG::computeRealResidual() {
    if (blockCountGPU != 0 && blockCountCPU != 0) {
        // exchange parts of x vector so that both CPU and GPU hold the complete vector
        gpuQueue.submit([&](handler& h) {
            h.memcpy(&x_gpu[blockStartCPU * conf::matrixBlockSize], &x[blockStartCPU * conf::matrixBlockSize],
                     blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
        gpuQueue.submit([&](handler& h) {
            h.memcpy(x.data(), x_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
    }
    waitAllQueues();


    // r = b - Ax
    if (blockCountGPU != 0) {
        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, x_gpu, r_gpu, 0, 0, blockCountGPU, A.blockCountXY,
                                                  A.blockCountXY);
    }
    if (blockCountCPU != 0) {
        MatrixVectorOperations::matrixVectorBlock_CPU(cpuQueue, A.matrixData.data(), x.data(), r_cpu, blockStartCPU, 0,
                                                  blockCountCPU, A.blockCountXY, A.blockCountXY);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::subVectorBlock(gpuQueue, b_gpu, r_gpu, r_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r_cpu, r_cpu, blockStartCPU,
                                         blockCountCPU);
    }
    waitAllQueues();
}

void CG::computeRealResidual_CommunicationHiding() {
    // if (blockCountGPU != 0 && blockCountCPU != 0) {
    //     // exchange parts of x vector so that both CPU and GPU hold the complete vector. Happens asynchronously.
    //     gpuQueue.submit([&](handler& h) {
    //         h.memcpy(&x_gpu[blockStartCPU * conf::matrixBlockSize], &x[blockStartCPU * conf::matrixBlockSize],
    //                  blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    //     });
    //     gpuQueue.submit([&](handler& h) {
    //         h.memcpy(x.data(), x_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
    //     });
    //
    //     // perform partial matrix-vector products with data that is already on the GPU / CPU
    // } else if (blockCountGPU != 0 && blockCountCPU == 0) {
    //     waitAllQueues();
    // } else if (blockCountCPU != 0 && blockCountGPU == 0) {
    //     waitAllQueues();
    // } else {
    //     throw std::runtime_error("Invalid CPU/GPU proportion");
    // }
}

void CG::update_r(conf::fp_type alpha) {
    // r = r - 𝛼q
    if (blockCountGPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(gpuQueue, r_gpu, -1.0 * alpha, q_gpu, r_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(cpuQueue, r_cpu, -1.0 * alpha, q_cpu, r_cpu, blockStartCPU,
                                                 blockCountCPU);
    }
    waitAllQueues();
}

void CG::compute_delta_new(conf::fp_type& delta_new) {
    unsigned int workGroupCountScalarProduct_GPU = 0;
    unsigned int workGroupCountScalarProduct_CPU = 0;

    // δ_new = r^T * r
    if (blockCountGPU != 0) {
        workGroupCountScalarProduct_GPU = VectorOperations::scalarProduct(
            gpuQueue, r_gpu, r_gpu, tmp_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        workGroupCountScalarProduct_CPU = VectorOperations::scalarProduct(
            cpuQueue, r_cpu, r_cpu, tmp_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::sumFinalScalarProduct(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    }
    waitAllQueues();

    // get value of δ_new from gpu
    delta_new = 0;
    if (blockCountGPU != 0) {
        gpuQueue.submit([&](handler& h) { h.memcpy(&delta_new, tmp_gpu, sizeof(conf::fp_type)); }).wait();
    }
    if (blockCountCPU != 0) {
        delta_new = delta_new + tmp_cpu[0];
    }
}

void CG::compute_d(conf::fp_type& beta) {
    // d = r + βd
    if (blockCountGPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(gpuQueue, r_gpu, beta, d_gpu, d_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(cpuQueue, r_cpu, beta, d_cpu, d_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();
}

void CG::waitAllQueues() {
    if (blockCountGPU != 0) {
        gpuQueue.wait();
    }
    if (blockCountCPU != 0) {
        cpuQueue.wait();
    }
}
