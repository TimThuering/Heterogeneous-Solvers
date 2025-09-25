#ifndef HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP
#define HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP

#include "Configuration.hpp"
#include "MetricsTracker.hpp"

/**
 * Abstract class from which all load balancer implementations are derived from
 */
class LoadBalancer {
public:
    LoadBalancer(int updateInterval, double initialProportionGPU, int blockCountXY);

    /**
     * Abstract method that gets implemented in child classes and determines a new load distribution between the GPU and CPU
     * @param metricsTracker A metrics tracker object that stores runtime metrices and can be used to calculate the new split.
     * @return the new GPU proportion
     */
    virtual double getNewProportionGPU(MetricsTracker &metricsTracker) = 0;

    int updateInterval; /// interval in which the split is updated

    double currentProportionGPU; /// current CPU / GPU split

    int blockCountXY; /// block count of the matrix A in x/y dimension

    virtual ~LoadBalancer() {}
};


#endif //HETEROGENEOUS_CONJUGATE_GRADIENTS_LOADBALANCER_HPP
