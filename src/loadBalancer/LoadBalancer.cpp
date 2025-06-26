#include "LoadBalancer.hpp"

LoadBalancer::LoadBalancer(int updateInterval, double initialProportionGPU, int blockCountXY) : updateInterval(updateInterval), currentProportionGPU(initialProportionGPU), blockCountXY(blockCountXY) {

}
