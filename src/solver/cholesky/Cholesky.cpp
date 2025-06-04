#include "Cholesky.hpp"

#include "MatrixMatrixOperations.hpp"
#include "MatrixOperations.hpp"
#include "MatrixParser.hpp"

Cholesky::Cholesky(SymmetricMatrix& A, queue& cpuQueue, queue& gpuQueue):
    A(A),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue) {
}

void Cholesky::solve() {
    // init GPU data structure for matrix A on which the Cholesky decomposition should be performed
    A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);

    // Copy the matrix A to the GPU
    gpuQueue.submit([&](handler& h) {
        h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
    }).wait();

    const int blockCountATotal = (A.blockCountXY * (A.blockCountXY + 1) / 2);


    // begin with tiled Cholesky decomposition using right-looking algorithm
    for (int k = 0; k < A.blockCountXY; ++k) {
    // for (int k = 0; k < 2; ++k) {
        std::cout << "Column: " << k << std::endl;
        // ID of diagonal block A_kk
        const int columnsToRight = A.blockCountXY - k;
        const int blockID = blockCountATotal - (columnsToRight * (columnsToRight + 1) / 2);

        // perform Cholesky decomposition on diagonal block A_kk
        MatrixOperations::cholesky(gpuQueue, A_gpu, blockID, k);
        gpuQueue.wait();

        if (k < A.blockCountXY - 1) {
            // solve triangular system for current column k below the diagonal
            MatrixMatrixOperations::triangularSolve(gpuQueue, A_gpu, blockID, k, k + 1,A.blockCountXY - (k + 1));
            gpuQueue.wait();

            // update the blocks on the diagonal below the current diagonal block
            MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedGPU(gpuQueue, A_gpu, blockID, k, k + 1,A.blockCountXY - (k + 1), A.blockCountXY);
            gpuQueue.wait();

            if (k < A.blockCountXY - 2) {
                // update the blocks int the lower triangle below the current diagonal block
                MatrixMatrixOperations::matrixMatrixStep_optimizedGPU(gpuQueue, A_gpu, blockID, k, k + 2,A.blockCountXY - (k + 2), A.blockCountXY);
                gpuQueue.wait();
            }
        }
    }



    gpuQueue.submit([&](handler& h) {
        h.memcpy(A.matrixData.data(), A_gpu, A.matrixData.size() * sizeof(conf::fp_type));
    }).wait();

    MatrixParser::writeFullMatrix("./A_chol_result", A);


}


// test code
// conf::fp_type* A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);
// for (int i = 0; i < 20; ++i) {
//     gpuQueue.submit([&](handler& h) {
//         h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
//     }).wait();
//     // MatrixOperations::cholesky_GPU_optimized(gpuQueue, A_gpu, 0,0);
//     // MatrixOperations::cholesky_GPU_optimized(cpuQueue, A.matrixData.data(), 0,0);
//     // gpuQueue.wait();
//
//     // sycl::event event = MatrixMatrixOperations::triangularSolve_optimizedGPU(gpuQueue, A_gpu, 0,0,1,A.blockCountXY -1);
//     // sycl::event event = MatrixMatrixOperations::triangularSolve(cpuQueue, A.matrixData.data(), 0,0, 1);
//     // gpuQueue.wait();
//
//     //        sycl::event event = MatrixMatrixOperations::symmetricMatrixMatrixDiagonal_optimizedGPU(gpuQueue, A_gpu, 0,0,1,A.blockCountXY -1, A.blockCountXY);
//     // sycl::event event = MatrixMatrixOperations::matrixMatrixStep(cpuQueue, A.matrixData.data(), 0,0,2,A.blockCountXY -2, A.blockCountXY);
//     sycl::event event = MatrixMatrixOperations::matrixMatrixStep_optimizedGPU(gpuQueue, A_gpu, 0,0,2,A.blockCountXY -2, A.blockCountXY);
//     //        sycl::event event = MatrixMatrixOperations::matrixMatrixStep(gpuQueue, A_gpu, 0,0,1,A.blockCountXY -2, A.blockCountXY);
//     gpuQueue.wait();
//     // cpuQueue.wait();
//
//
//     std::cout << static_cast<double>(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
//         event.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6 << std::endl;
// }
//
//
// gpuQueue.submit([&](handler& h) {
//     h.memcpy(A.matrixData.data(), A_gpu, A.matrixData.size() * sizeof(conf::fp_type));
// }).wait();

// MatrixParser::writeFullMatrix("./A_chol_test", A);
