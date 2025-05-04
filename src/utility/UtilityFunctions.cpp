#include "UtilityFunctions.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

void UtilityFunctions::writeResult(const std::string &path, const std::vector<double, sycl::usm_allocator<double, sycl::usm::alloc::shared>> &x) {
    std::string filePath = path + "/x_result.txt";
    std::ofstream output(filePath);

    output << std::setprecision(20) << std::fixed;

    for (auto &element: x) {
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
