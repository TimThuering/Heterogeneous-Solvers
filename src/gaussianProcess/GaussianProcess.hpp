#ifndef GAUSSIANPROCESS_HPP
#define GAUSSIANPROCESS_HPP
#include "LoadBalancer.hpp"
#include "RightHandSide.hpp"
#include "SymmetricMatrix.hpp"

/**
 * This class implements the Gaussian Process pipeline for prediction
 */
class GaussianProcess {
public:
    GaussianProcess(SymmetricMatrix& A, RightHandSide& train_y, std::string& path_train, std::string& path_test, sycl::queue& cpuQueue, sycl::queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrix& A; /// training-training kernel


    RightHandSide& train_y; /// training targets


    std::string& path_train; /// path to training data
    std::string& path_test; /// path to test data


    sycl::queue& cpuQueue; /// SYCL CPU queue
    sycl::queue& gpuQueue; /// SYCL GPU queue
    std::shared_ptr<LoadBalancer> loadBalancer;


    /// starts the computation for the Gaussian Processes Prediction
    void start();

private:
};


#endif //GAUSSIANPROCESS_HPP
