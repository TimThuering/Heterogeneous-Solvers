#include "UtilityFunctions.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <hws/cpu/hardware_sampler.hpp>

#include "Configuration.hpp"
#include "MatrixGenerator.hpp"
#include "MatrixVectorOperations.hpp"
#include "RightHandSide.hpp"
#include "SymmetricMatrix.hpp"

void UtilityFunctions::writeResult(const std::string& path,
                                   const std::vector<
                                       conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>>&
                                   x) {
    std::string filePath = path + "/x_result.txt";
    std::ofstream output(filePath);

    output << std::setprecision(20) << std::fixed;

    for (auto& element : x) {
        output << element << " ";
    }
    output << std::endl;
    output.close();
}

std::string UtilityFunctions::getTimeString() {
    auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto timeStruct = *std::localtime(&currentTime);
    std::ostringstream timeStringStream;
    timeStringStream << std::put_time(&timeStruct, "%Y_%m_%d__%H_%M_%S");
    std::string timeString = timeStringStream.str();
    return timeString;
}

void UtilityFunctions::measureIdlePowerCPU() {
    hws::cpu_hardware_sampler cpuSampler{hws::sample_category::power};

    cpuSampler.start_sampling();
    sleep(std::ceil(10 * static_cast<double>(cpuSampler.sampling_interval().count()) / 1000.0));
    cpuSampler.stop_sampling();


    if (cpuSampler.power_samples().get_power_usage().has_value()) {
        std::vector<double> idlePower = cpuSampler.power_samples().get_power_usage().value();
        const double idleWatt_CPU =
            std::accumulate(idlePower.begin(), idlePower.end(), 0.0) / static_cast<double>(idlePower.size());
        conf::idleWatt_CPU = idleWatt_CPU;
        std::cout << "-- Setting CPU idle power to " <<  conf::idleWatt_CPU << "W." << std::endl;
    } else {
        std::cerr << "\033[93m[WARNING]\033[0m Could not determine CPU idle power draw. Using default of " <<
            conf::idleWatt_CPU << "W." << std::endl;
    }
}

double UtilityFunctions::checkResult(RightHandSide& b, sycl::queue cpuQueue, sycl::queue gpuQueue, std::string path_gp_input, std::string path_gp_output) {
    std::cout << "Checking result" << std::endl;
    SymmetricMatrix A_new = MatrixGenerator::generateSPDMatrix(path_gp_input, cpuQueue, gpuQueue);

    RightHandSide result(conf::N, conf::matrixBlockSize, gpuQueue);

#if USE_DPCPP
    cpuQueue = gpuQueue;
#endif

    MatrixVectorOperations::matrixVectorBlock(cpuQueue,A_new.matrixData.data(),b.rightHandSideData.data(),result.rightHandSideData.data(),0,0,b.blockCountX,b.blockCountX,A_new.blockCountXY,true);
    cpuQueue.wait();
    RightHandSide b_new = MatrixGenerator::parseRHS_GP(path_gp_output, gpuQueue);

    double error = 0.0;
    for (unsigned int i = 0; i < b.rightHandSideData.size(); i++) {
        error += std::abs(b_new.rightHandSideData[i] - result.rightHandSideData[i]);
    }
    error /= static_cast<double>(conf::N);

    return error;
}