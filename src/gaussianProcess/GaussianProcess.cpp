#include "GaussianProcess.hpp"

#include "CG.hpp"
#include "MatrixGenerator.hpp"
#include "MatrixVectorOperations.hpp"
#include "UtilityFunctions.hpp"
#include "cholesky/Cholesky.hpp"
#include "cholesky/TriangularSystemSolver.hpp"

GaussianProcess::GaussianProcess(SymmetricMatrix& A, RightHandSide& train_y, std::string& path_train, std::string& path_test, sycl::queue& cpuQueue, sycl::queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer) :
    A(A),
    train_y(train_y),
    path_train(path_train),
    path_test(path_test),
    cpuQueue(cpuQueue),
    gpuQueue(gpuQueue),
    loadBalancer(std::move(loadBalancer)) {
}

void GaussianProcess::start() {
    if (conf::algorithm == "cg") {
        CG cg(A, train_y, cpuQueue, gpuQueue, loadBalancer);
        cg.solveHeterogeneous();
        train_y.rightHandSideData.assign(cg.x.begin(), cg.x.end());
    } else if (conf::algorithm == "cholesky") {
        Cholesky cholesky(A, cpuQueue, gpuQueue, loadBalancer);
        cholesky.solve_heterogeneous();
        TriangularSystemSolver solver(A, cholesky.A_gpu, train_y, cpuQueue, gpuQueue, loadBalancer);
        double solveTime = solver.solve();
        if (conf::trackCholeskySolveStep) {
            if (conf::printVerbose && conf::enableHWS) {
                std::cout << "Ending tracking after solve step" << std::endl;
            }
            cholesky.metricsTracker.endTracking();
        }
        cholesky.metricsTracker.solveTime = solveTime;
        cholesky.writeMetricsToFile();
    } else {
        throw std::runtime_error("Invalid algorithm: " + conf::algorithm);
    }

    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> K_star{
        usm_allocator<conf::fp_type, usm::alloc::host>(cpuQueue)
    };
    K_star.resize(conf::N * conf::N_test);

    MatrixGenerator::generateTestKernelMatrix(path_train, path_test, cpuQueue, gpuQueue, K_star.data());

    std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> result{
        usm_allocator<conf::fp_type, usm::alloc::host>(cpuQueue)
    };
    result.resize(conf::N_test);

    MatrixVectorOperations::matrixVectorGP(cpuQueue, K_star.data(), train_y.rightHandSideData.data(), result.data(), conf::N, conf::N_test);


    if (conf::writeResult) {
        UtilityFunctions::writeResult(".", result);
    }
}
