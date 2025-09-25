#ifndef POWERLOADBALANCER_HPP
#define POWERLOADBALANCER_HPP
#include "LoadBalancer.hpp"

/**
 * This class contains a load balancer implementation that determines the GPU proportion based on power metrics.
 */
class PowerLoadBalancer : public LoadBalancer {
public:
    PowerLoadBalancer(int updateInterval, double initialProportionGPU, int blockCountXY);


    /**
     * Determines the new GPU proportion based on the power metrics obtained int the last update interval.
     *
     * @param metricsTracker metrics tracker for power metrics (among others)
     * @return the new GPU proportion
     */
    double getNewProportionGPU(MetricsTracker &metricsTracker) override;
};


#endif //POWERLOADBALANCER_HPP
