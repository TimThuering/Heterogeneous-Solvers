#ifndef POWERLOADBALANCER_HPP
#define POWERLOADBALANCER_HPP
#include "LoadBalancer.hpp"


class PowerLoadBalancer : public LoadBalancer {
public:
    PowerLoadBalancer(int updateInterval, double initialProportionGPU);


    double getNewProportionGPU(MetricsTracker &metricsTracker) override;
};


#endif //POWERLOADBALANCER_HPP
