#ifndef MATRIXGENERATOR_HPP
#define MATRIXGENERATOR_HPP

#include <sycl/sycl.hpp>

#include "RightHandSide.hpp"
#include "SymmetricMatrix.hpp"

/**
 * Class that contains various methods for generating symmetric positive-definite (SPD) matrices
 */
class MatrixGenerator {
public:
    /**
     * This operation generates a diagonal dominant SPD matrix. (Currently not used)
     *
     * @param queue SYCL queue
     * @return a SPD matrix
     */
    static SymmetricMatrix generateSPDMatrixStrictDiagonalDominant(sycl::queue& queue);

    /**
     * This operation generates a SPD kernel matrix that can be used for Gaussian Processes
     *
     * @param path path to text file with data
     * @param queue CPU queue
     * @param queueGPU GPU queue
     * @return a SPD matrix
     */
    static SymmetricMatrix generateSPDMatrix(std::string& path, sycl::queue& queue, sycl::queue& queueGPU);

    /**
     * This operation generates the test kernel matrix K* used required for prediction with Gaussian Processes.
     *
     * @param path_train path to training data
     * @param path_test path to test data
     * @param queue CPU queue
     * @param queueGPU GPU queue
     * @param K_star the test kernel matrix
     */
    static void generateTestKernelMatrix(std::string& path_train, std::string& path_test, sycl::queue& queue, sycl::queue& queueGPU, conf::fp_type* K_star);

    /**
     * This operation parses the right hand side based on data for Gaussian processses
     *
     * @param path path to right hand side data
     * @param queue SYCL queue
     * @return the right hand side
     */
    static RightHandSide parseRHS_GP(std::string& path, sycl::queue& queue);

    /**
     * This operation generates a right hand side randomly
     *
     * @param queue SYCL queue
     * @return the right hand side
     */
    static RightHandSide generateRHS(sycl::queue& queue);

private:
    static void readInputVector(std::string& path,std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>>& dataVector, int N, int offset);
};


#endif //MATRIXGENERATOR_HPP
