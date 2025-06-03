#include "Cholesky.hpp"

#include "MatrixMatrixOperations.hpp"

Cholesky::Cholesky(SymmetricMatrix& A, queue& cpuQueue, queue& gpuQueue):
    A(A),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue) {
}

void Cholesky::solve() {

    // test code
    conf::fp_type* A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);
    for (int i = 0; i < 20; ++i) {
        gpuQueue.submit([&](handler& h) {
            h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
        }).wait();
        // MatrixOperations::cholesky_GPU_optimized(gpuQueue, A_gpu, 0,0);
        // MatrixOperations::cholesky_GPU_optimized(cpuQueue, A.matrixData.data(), 0,0);
        // gpuQueue.wait();

        // sycl::event event = MatrixMatrixOperations::triangularSolve_optimizedGPU(gpuQueue, A_gpu, 0,0,1,A.blockCountXY -1);
        // sycl::event event = MatrixMatrixOperations::triangularSolve(cpuQueue, A.matrixData.data(), 0,0, 1);
        // gpuQueue.wait();

        //        sycl::event event = MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedGPU(gpuQueue, A_gpu, 0,0,1,A.blockCountXY -1, A.blockCountXY);
        // sycl::event event = MatrixMatrixOperations::matrixMatrixStep(cpuQueue, A.matrixData.data(), 0,0,2,A.blockCountXY -2, A.blockCountXY);
        sycl::event event = MatrixMatrixOperations::matrixMatrixStep_optimizedGPU(gpuQueue, A_gpu, 0,0,2,A.blockCountXY -2, A.blockCountXY);
        //        sycl::event event = MatrixMatrixOperations::matrixMatrixStep(gpuQueue, A_gpu, 0,0,1,A.blockCountXY -2, A.blockCountXY);
        gpuQueue.wait();
        // cpuQueue.wait();


        std::cout << static_cast<double>(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
            event.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6 << std::endl;
    }


    gpuQueue.submit([&](handler& h) {
        h.memcpy(A.matrixData.data(), A_gpu, A.matrixData.size() * sizeof(conf::fp_type));
    }).wait();

    // MatrixParser::writeFullMatrix("./A_chol_test", A);
}
