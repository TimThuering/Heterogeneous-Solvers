#ifndef GAUSSIANPROCESS_HPP
#define GAUSSIANPROCESS_HPP
#include "LoadBalancer.hpp"
#include "RightHandSide.hpp"
#include "SymmetricMatrix.hpp"


class GaussianProcess {
public:
    GaussianProcess(SymmetricMatrix& A,  RightHandSide& train_y, std::string& path_train, std::string& path_test, sycl::queue& cpuQueue, sycl::queue& gpuQueue, std::shared_ptr<LoadBalancer> loadBalancer);

    // training-training kernel
    SymmetricMatrix& A;

    // training targets
    RightHandSide& train_y;


    std::string& path_train;
    std::string& path_test;



    sycl::queue& cpuQueue;
    sycl::queue& gpuQueue;
    std::shared_ptr<LoadBalancer> loadBalancer;


    void start();

private:

};



#endif //GAUSSIANPROCESS_HPP
