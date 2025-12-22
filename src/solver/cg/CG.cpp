#include "CG.hpp"

#include <iostream>
#include <sycl/sycl.hpp>

#include "MatrixParser.hpp"
#include "MatrixVectorOperations.hpp"
#include "VectorOperations.hpp"
#include "UtilityFunctions.hpp"

using namespace sycl;

CG::CG(SymmetricMatrix &A, RightHandSide &b, queue &cpuQueue, queue &gpuQueue,
       std::shared_ptr<LoadBalancer> loadBalancer) : A(A),
                                                     b(b),
                                                     x(sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>(gpuQueue)),
                                                     cpuQueue(cpuQueue),
                                                     gpuQueue(gpuQueue),
                                                     loadBalancer(std::move(loadBalancer)) {
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

void CG::solveHeterogeneous() {
    const auto start = std::chrono::steady_clock::now();
    metricsTracker.startTracking();

    // get new GPU proportion of workload
    double gpuProportion = loadBalancer->currentProportionGPU;
    blockCountGPU = std::ceil(static_cast<double>(A.blockCountXY) * gpuProportion);
    blockCountCPU = A.blockCountXY - blockCountGPU;
    blockStartCPU = blockCountGPU;

    // initialize data structures
    auto startMemInit = std::chrono::steady_clock::now();
    initCPUdataStructures();
    initGPUdataStructures();
    auto endMemInit = std::chrono::steady_clock::now();
    metricsTracker.memoryInitTime = std::chrono::duration<double, std::milli>(endMemInit - startMemInit).count();

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


    bool firstIteration = true;

    while (iteration < conf::iMax && delta_new > epsilon2 * delta_zero) {
        auto startIteration = std::chrono::steady_clock::now();

        if (iteration % loadBalancer->updateInterval == 0 && !firstIteration) {
            rebalanceProportions(gpuProportion);
        }

        auto timePoint1 = std::chrono::steady_clock::now();
        compute_q(); // q = Ad

        auto timePoint2 = std::chrono::steady_clock::now();
        compute_alpha(alpha, delta_new); // 𝛼 = δ_new / d^T * q

        auto timePoint3 = std::chrono::steady_clock::now();
        update_x(alpha); // x = x + 𝛼d

        auto timePoint4 = std::chrono::steady_clock::now();
        if (iteration % 50 == 0) {
            // compute real residual every 50 iterations --> requires additional matrix vector product
            computeRealResidual(); // r = b - Ax
        } else {
            // compute residual without an additional matrix vector product
            update_r(alpha); // r = r - 𝛼q
        }

        auto timePoint5 = std::chrono::steady_clock::now();
        delta_old = delta_new; // δ_old = δ_new
        compute_delta_new(delta_new); // δ_new = r^T * r
        auto timePoint6 = std::chrono::steady_clock::now();
        beta = delta_new / delta_old; // β = δ_new / δ_old
        compute_d(beta); // d = r + βd

        auto endIteration = std::chrono::steady_clock::now();
        auto iterationTime = std::chrono::duration<double, std::milli>(endIteration - startIteration).count();
        metricsTracker.updateMetrics(iteration, blockCountGPU, blockCountCPU, iterationTime, loadBalancer->updateInterval);
        if (conf::printVerbose) {
            std::cout << iteration << ": Iteration time: " << iterationTime << "ms" << std::endl;
        }

        auto time_q = std::chrono::duration<double, std::milli>(timePoint2 - timePoint1).count();
        auto time_alpha = std::chrono::duration<double, std::milli>(timePoint3 - timePoint2).count();
        auto time_x = std::chrono::duration<double, std::milli>(timePoint4 - timePoint3).count();
        auto time_r = std::chrono::duration<double, std::milli>(timePoint5 - timePoint4).count();
        auto time_delta = std::chrono::duration<double, std::milli>(timePoint6 - timePoint5).count();
        auto time_d = std::chrono::duration<double, std::milli>(endIteration - timePoint6).count();

        metricsTracker.times_q.push_back(time_q);
        metricsTracker.times_alpha.push_back(time_alpha);
        metricsTracker.times_x.push_back(time_x);
        metricsTracker.times_r.push_back(time_r);
        metricsTracker.times_delta.push_back(time_delta);
        metricsTracker.times_d.push_back(time_d);


        iteration++;
        firstIteration = false;
    }
    waitAllQueues();

    if (blockCountGPU != 0 && conf::unifiedAddressSpace == false) {
        auto startMemCopy = std::chrono::steady_clock::now();
        gpuQueue.submit([&](handler &h) {
            h.memcpy(x.data(), x_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        }).wait();
        auto endMemCopy = std::chrono::steady_clock::now();
        metricsTracker.resultCopyTime = std::chrono::duration<double, std::milli>(endMemCopy - startMemCopy).count();
    }

    auto end = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << totalTime << "ms (" << iteration << " iterations)" << std::endl;
    std::cout << "Memory init: " << metricsTracker.memoryInitTime << "ms" << std::endl;
    std::cout << "Result copy: " << metricsTracker.resultCopyTime << "ms" << std::endl;
    metricsTracker.totalTime = totalTime;
    metricsTracker.endTracking();

    std::string timeString = UtilityFunctions::getTimeString();
    std::string filePath = conf::outputPath + "/" + timeString;
    std::filesystem::create_directories(filePath);
    metricsTracker.writeJSON(filePath);
    if (conf::writeResult) {
        UtilityFunctions::writeResult(".", x);
    }

    freeDataStructures();
}

void CG::initGPUdataStructures() {
    if (blockCountGPU == 0) {
        return;
    }

    const std::size_t gpuAvailableMemorySize = 0.9 * gpuQueue.get_device().get_info<sycl::info::device::global_mem_size>() - 6 * b.rightHandSideData.size() * sizeof(conf::fp_type);

    const double maxBlocksGPUMemory = std::floor(static_cast<double>(gpuAvailableMemorySize) / (static_cast<double>(conf::matrixBlockSize) * static_cast<double>(conf::matrixBlockSize) * sizeof(conf::fp_type)));

    const int totalBlockCountA = (A.blockCountXY * (A.blockCountXY + 1) / 2);

    std::size_t valuesGPU;
    if (maxBlocksGPUMemory < totalBlockCountA) {
        // GPU memory is not sufficient to store the whole matrix --> calculate maximum number of rows that can be stored

        const double valueRoot = (2 * A.blockCountXY + 1) * (2 * A.blockCountXY + 1) - 8.0 * maxBlocksGPUMemory;
        if (valueRoot >= 0.0) {
            const std::size_t maxRowsGPUMemory = std::floor(0.5 + A.blockCountXY - 0.5 * std::sqrt(valueRoot));

            const std::size_t rowsLowerPart = A.blockCountXY - maxRowsGPUMemory;

            const std::size_t blocksGPUMemory = totalBlockCountA - (rowsLowerPart * (rowsLowerPart + 1) / 2);

            valuesGPU = blocksGPUMemory * conf::matrixBlockSize * conf::matrixBlockSize;

            maxBlockCountGPU = maxRowsGPUMemory;

            if (maxRowsGPUMemory < blockCountGPU || blockCountCPU == 0) {
                throw std::runtime_error("GPU memory not sufficient for requested split between GPU and CPU");
            }
        } else {
            throw std::runtime_error("Error during GPU memory allocation. Choose a different Block size.");
        }
    } else {
        maxBlockCountGPU = A.blockCountXY;
        // Whole matrix A fits into GPU memory
        valuesGPU = A.matrixData.size();
    }

    if (conf::unifiedAddressSpace == false) {
        // Matrix A GPU
        A_gpu = malloc_device<conf::fp_type>(valuesGPU, gpuQueue);
        gpuQueue.submit([&](handler &h) {
            h.memcpy(A_gpu, A.matrixData.data(), valuesGPU * sizeof(conf::fp_type));
        }).wait();

        // Right-hand side b GPU
        b_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);
        gpuQueue.submit([&](handler &h) {
            h.memcpy(b_gpu, b.rightHandSideData.data(), b.rightHandSideData.size() * sizeof(conf::fp_type));
        }).wait();

        // result vector x
        x_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

        // residual vector r
        r_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

        // vector d
        d_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

        // vector q
        q_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

        // temporary vector
        tmp_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);
    } else {
        // data structures can now be shared between CPU and GPU
        A_gpu = A.matrixData.data();

        b_gpu = b.rightHandSideData.data();

        x_gpu = x.data();

        r_gpu = r_cpu;

        d_gpu = d_cpu;

        q_gpu = q_cpu;

        // GPU needs to have its own tmp vector
        tmp_gpu = malloc_device<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);
    }

    if (A_gpu == nullptr || b_gpu == nullptr || x_gpu == nullptr || r_gpu == nullptr || d_gpu == nullptr || q_gpu == nullptr || tmp_gpu == nullptr) {
        throw std::runtime_error("Error during GPU memory allocation");
    }
}

void CG::initCPUdataStructures() {
    if (blockCountCPU == 0 && conf::unifiedAddressSpace == false) {
        return;
    }

    // residual vector r
    r_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // vector d
    d_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // vector q
    q_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    // temporary vector
    tmp_cpu = malloc_host<conf::fp_type>(b.rightHandSideData.size(), gpuQueue);

    if (r_cpu == nullptr || d_cpu == nullptr || q_cpu == nullptr || tmp_cpu == nullptr) {
        throw std::runtime_error("Error during CPU memory allocation");
    }
}

void CG::freeDataStructures() {
    if (blockCountGPU != 0 && conf::unifiedAddressSpace == false) {
        sycl::free(A_gpu, gpuQueue);
        sycl::free(b_gpu, gpuQueue);
        sycl::free(x_gpu, gpuQueue);
        sycl::free(r_gpu, gpuQueue);
        sycl::free(d_gpu, gpuQueue);
        sycl::free(q_gpu, gpuQueue);
        sycl::free(tmp_gpu, gpuQueue);
    }

    if (blockCountCPU != 0) {
        sycl::free(r_cpu, gpuQueue);
        sycl::free(d_cpu, gpuQueue);
        sycl::free(q_cpu, gpuQueue);
        sycl::free(tmp_cpu, gpuQueue);
    }
}

void CG::initCG(conf::fp_type &delta_zero, conf::fp_type &delta_new) {
    // r = b - Ax
    if (blockCountGPU != 0) {
        MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, x_gpu, r_gpu, 0, 0, blockCountGPU, A.blockCountXY, A.blockCountXY);
    }
    if (blockCountCPU != 0) {
        MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), x.data(), r_cpu, blockStartCPU, 0, blockCountCPU, A.blockCountXY, A.blockCountXY);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::subVectorBlock(gpuQueue, b_gpu, r_gpu, r_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r_cpu, r_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();

    // d = r
    if (blockCountGPU != 0) {
        gpuQueue.submit([&](handler &h) {
            h.memcpy(d_gpu, r_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
    }
    if (blockCountCPU != 0) {
        cpuQueue.submit([&](handler &h) {
            h.memcpy(&d_cpu[blockStartCPU * conf::matrixBlockSize], &r_cpu[blockStartCPU * conf::matrixBlockSize], blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
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
                VectorOperations::scalarProduct_CPU(cpuQueue, r_cpu, r_cpu, tmp_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::sumFinalScalarProduct_CPU(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    }
    waitAllQueues();


    delta_new = 0;
    if (blockCountGPU != 0) {
        // get value of δ_new from gpu
        gpuQueue.submit([&](handler &h) { h.memcpy(&delta_new, tmp_gpu, sizeof(conf::fp_type)); }).wait();
    }
    if (blockCountCPU != 0) {
        delta_new = delta_new + tmp_cpu[0];
    }
    delta_zero = delta_new;
}

void CG::compute_q() {
    auto startmemcpy = std::chrono::steady_clock::now();

    if ((blockCountGPU != 0 && blockCountCPU != 0) && conf::unifiedAddressSpace == false) {
        // exchange parts of d vector so that both CPU and GPU hold the complete vector
        gpuQueue.submit([&](handler &h) {
            h.memcpy(&d_gpu[blockStartCPU * conf::matrixBlockSize], &d_cpu[blockStartCPU * conf::matrixBlockSize], blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
        gpuQueue.submit([&](handler &h) {
            h.memcpy(d_cpu, d_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
    }
    waitAllQueues();
    auto endmemcpy = std::chrono::steady_clock::now();

    auto memcpyTime = std::chrono::duration<double, std::milli>(endmemcpy - startmemcpy).count();
    metricsTracker.memcopy_d.push_back(memcpyTime);


    sycl::event eventGPU;
    sycl::event eventCPU;
    // q = Ad
    if (blockCountGPU != 0) {
        if (conf::gpuOptimizationLevel == 0) {
            eventGPU = MatrixVectorOperations::matrixVectorBlock(gpuQueue, A_gpu, d_gpu, q_gpu, 0, 0, blockCountGPU, A.blockCountXY, A.blockCountXY);
        } else {
            eventGPU = MatrixVectorOperations::matrixVectorBlock_GPU(gpuQueue, A_gpu, d_gpu, q_gpu, 0, 0, blockCountGPU, A.blockCountXY, A.blockCountXY);
        }
    }
    if (blockCountCPU != 0) {
        if (conf::cpuOptimizationLevel == 0) {
            eventCPU = MatrixVectorOperations::matrixVectorBlock(cpuQueue, A.matrixData.data(), d_cpu, q_cpu, blockStartCPU, 0, blockCountCPU, A.blockCountXY, A.blockCountXY);
        } else {
            eventCPU = MatrixVectorOperations::matrixVectorBlock_CPU(cpuQueue, A.matrixData.data(), d_cpu, q_cpu, blockStartCPU, 0, blockCountCPU, A.blockCountXY, A.blockCountXY);
        }
    }
    waitAllQueues();

    bool intel = false;
#ifdef INTEL
    intel = true;
#endif

    // append execution times
    if (blockCountGPU != 0 && !intel) {
        metricsTracker.matrixVectorTimes_GPU.push_back(
            static_cast<double>(eventGPU.get_profiling_info<sycl::info::event_profiling::command_end>() - eventGPU.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6);
    } else {
        metricsTracker.matrixVectorTimes_GPU.push_back(0);
    }
    if (blockCountCPU != 0 && !intel) {
        metricsTracker.matrixVectorTimes_CPU.push_back(
            static_cast<double>(eventCPU.get_profiling_info<sycl::info::event_profiling::command_end>() - eventCPU.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6);
    } else {
        metricsTracker.matrixVectorTimes_CPU.push_back(0);
    }
}

void CG::compute_alpha(conf::fp_type &alpha, conf::fp_type &delta_new) {
    unsigned int workGroupCountScalarProduct_GPU = 0;
    unsigned int workGroupCountScalarProduct_CPU = 0;

    // 𝛼 = δ_new / d^T * q
    if (blockCountGPU != 0) {
        workGroupCountScalarProduct_GPU = VectorOperations::scalarProduct(gpuQueue, d_gpu, q_gpu, tmp_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        workGroupCountScalarProduct_CPU = VectorOperations::scalarProduct_CPU(cpuQueue, d_cpu, q_cpu, tmp_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::sumFinalScalarProduct_CPU(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    }
    waitAllQueues();


    conf::fp_type result = 0;
    if (blockCountGPU != 0) {
        gpuQueue.submit([&](handler &h) { h.memcpy(&result, tmp_gpu, sizeof(conf::fp_type)); }).wait();
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
        VectorOperations::scaleAndAddVectorBlock(cpuQueue, x.data(), alpha, d_cpu, x.data(), blockStartCPU, blockCountCPU);
    }
    waitAllQueues();
}

void CG::computeRealResidual() {
    if ((blockCountGPU != 0 && blockCountCPU != 0) && conf::unifiedAddressSpace == false) {
        // exchange parts of x vector so that both CPU and GPU hold the complete vector
        gpuQueue.submit([&](handler &h) {
            h.memcpy(&x_gpu[blockStartCPU * conf::matrixBlockSize], &x[blockStartCPU * conf::matrixBlockSize], blockCountCPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
        gpuQueue.submit([&](handler &h) {
            h.memcpy(x.data(), x_gpu, blockCountGPU * conf::matrixBlockSize * sizeof(conf::fp_type));
        });
    }
    waitAllQueues();


    // r = b - Ax
    if (blockCountGPU != 0) {
        MatrixVectorOperations::matrixVectorBlock_GPU(gpuQueue, A_gpu, x_gpu, r_gpu, 0, 0, blockCountGPU, A.blockCountXY, A.blockCountXY);
    }
    if (blockCountCPU != 0) {
        MatrixVectorOperations::matrixVectorBlock_CPU(cpuQueue, A.matrixData.data(), x.data(), r_cpu, blockStartCPU, 0, blockCountCPU, A.blockCountXY, A.blockCountXY);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::subVectorBlock(gpuQueue, b_gpu, r_gpu, r_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::subVectorBlock(cpuQueue, b.rightHandSideData.data(), r_cpu, r_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();
}

void CG::update_r(conf::fp_type alpha) {
    // r = r - 𝛼q
    if (blockCountGPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(gpuQueue, r_gpu, -1.0 * alpha, q_gpu, r_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::scaleAndAddVectorBlock(cpuQueue, r_cpu, -1.0 * alpha, q_cpu, r_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();
}

void CG::compute_delta_new(conf::fp_type &delta_new) {
    unsigned int workGroupCountScalarProduct_GPU = 0;
    unsigned int workGroupCountScalarProduct_CPU = 0;

    // δ_new = r^T * r
    if (blockCountGPU != 0) {
        workGroupCountScalarProduct_GPU = VectorOperations::scalarProduct(gpuQueue, r_gpu, r_gpu, tmp_gpu, 0, blockCountGPU);
    }
    if (blockCountCPU != 0) {
        workGroupCountScalarProduct_CPU = VectorOperations::scalarProduct_CPU(cpuQueue, r_cpu, r_cpu, tmp_cpu, blockStartCPU, blockCountCPU);
    }
    waitAllQueues();

    if (blockCountGPU != 0) {
        VectorOperations::sumFinalScalarProduct(gpuQueue, tmp_gpu, workGroupCountScalarProduct_GPU);
    }
    if (blockCountCPU != 0) {
        VectorOperations::sumFinalScalarProduct_CPU(cpuQueue, tmp_cpu, workGroupCountScalarProduct_CPU);
    }
    waitAllQueues();

    // get value of δ_new from gpu
    delta_new = 0;
    if (blockCountGPU != 0) {
        gpuQueue.submit([&](handler &h) { h.memcpy(&delta_new, tmp_gpu, sizeof(conf::fp_type)); }).wait();
    }
    if (blockCountCPU != 0) {
        delta_new = delta_new + tmp_cpu[0];
    }
}

void CG::compute_d(conf::fp_type &beta) {
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

void CG::rebalanceProportions(double &gpuProportion) {
    // get new GPU proportion of workload
    gpuProportion = loadBalancer->getNewProportionGPU(metricsTracker);
    std::size_t blockCountGPU_new = std::ceil(static_cast<double>(A.blockCountXY) * gpuProportion);
    std::size_t blockCountCPU_new = A.blockCountXY - blockCountGPU_new;
    std::size_t blockStartCPU_new = blockCountGPU_new;

    if (blockCountGPU_new > maxBlockCountGPU) {
        if (conf::printVerbose) {
            std::cout << "Change in block counts would result into too much gpu memory usage. New GPU block count: " << maxBlockCountGPU << std::endl;
        }

        blockCountGPU_new = maxBlockCountGPU;
        blockCountCPU_new = A.blockCountXY - maxBlockCountGPU;
        blockStartCPU_new = maxBlockCountGPU;
    }

    if (blockCountGPU_new > blockCountGPU + conf::blockUpdateThreshold || blockCountCPU_new > blockCountCPU + conf::blockUpdateThreshold) {
        if (blockCountGPU_new > blockCountGPU) {
            const std::size_t additionalBlocks = blockCountGPU_new - blockCountGPU;
            // exchange missing parts of d vector
            gpuQueue.submit([&](handler &h) {
                h.memcpy(&d_gpu[blockStartCPU * conf::matrixBlockSize], &d_cpu[blockStartCPU * conf::matrixBlockSize], additionalBlocks * conf::matrixBlockSize * sizeof(conf::fp_type));
            });

            // exchange missing parts of x vector
            gpuQueue.submit([&](handler &h) {
                h.memcpy(&x_gpu[blockStartCPU * conf::matrixBlockSize], &x[blockStartCPU * conf::matrixBlockSize], additionalBlocks * conf::matrixBlockSize * sizeof(conf::fp_type));
            });

            // exchange missing parts of r vector
            gpuQueue.submit([&](handler &h) {
                h.memcpy(&r_gpu[blockStartCPU * conf::matrixBlockSize], &r_cpu[blockStartCPU * conf::matrixBlockSize], additionalBlocks * conf::matrixBlockSize * sizeof(conf::fp_type));
            });
        } else if (blockCountCPU_new > blockCountCPU) {
            const std::size_t additionalBlocks = blockCountCPU_new - blockCountCPU;

            // exchange missing parts of d vector
            gpuQueue.submit([&](handler &h) {
                h.memcpy(&d_cpu[blockStartCPU_new * conf::matrixBlockSize], &d_gpu[blockStartCPU_new * conf::matrixBlockSize], additionalBlocks * conf::matrixBlockSize * sizeof(conf::fp_type));
            });

            // exchange missing parts of x vector
            gpuQueue.submit([&](handler &h) {
                h.memcpy(&x[blockStartCPU_new * conf::matrixBlockSize], &x_gpu[blockStartCPU_new * conf::matrixBlockSize], additionalBlocks * conf::matrixBlockSize * sizeof(conf::fp_type));
            });

            // exchange missing parts of r vector
            gpuQueue.submit([&](handler &h) {
                h.memcpy(&r_cpu[blockStartCPU_new * conf::matrixBlockSize], &r_gpu[blockStartCPU_new * conf::matrixBlockSize], additionalBlocks * conf::matrixBlockSize * sizeof(conf::fp_type));
            });
        }

        waitAllQueues();

        blockCountGPU = blockCountGPU_new;
        blockCountCPU = blockCountCPU_new;
        blockStartCPU = blockStartCPU_new;
    } else if (blockCountGPU_new != blockCountGPU) {
        if (conf::printVerbose) {
            std::cout << "Change in block counts smaller than threshold --> no re-balancing: " << blockCountGPU << " --> " << blockCountGPU_new << std::endl;
        }
    }

    if (conf::printVerbose) {
        std::cout << "Block count GPU: " << blockCountGPU << std::endl;
        std::cout << "Block count CPU: " << blockCountCPU << std::endl;
        std::cout << "New GPU proportion: " << gpuProportion * 100 << "%" << std::endl;
    }
}
