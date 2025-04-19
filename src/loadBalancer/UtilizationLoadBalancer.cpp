#include "UtilizationLoadBalancer.hpp"

UtilizationLoadBalancer::UtilizationLoadBalancer(int updateInterval) : LoadBalancer(updateInterval),
                                                                       cpuUtilSampler(hws::sample_category::general) {

}
