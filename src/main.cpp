#include <iostream>
#include <sycl/sycl.hpp>
#include <hws/system_hardware_sampler.hpp>
#include <hws/cpu/hardware_sampler.hpp>
#include <hws/gpu_nvidia/hardware_sampler.hpp>
#include <hws/gpu_amd/hardware_sampler.hpp>
#include <cxxopts.hpp>

#include "MatrixParser.hpp"
#include "SymmetricMatrix.hpp"
#include "CG.hpp"
#include "LoadBalancer.hpp"
#include "StaticLoadBalancer.hpp"
#include "UtilizationLoadBalancer.hpp"
#include "RuntimeLoadBalancer.hpp"
#include "PowerLoadBalancer.hpp"
#include "MatrixGenerator.hpp"
#include "MatrixMatrixOperations.hpp"
#include "UtilityFunctions.hpp"
#include "MatrixOperations.hpp"

using namespace sycl;

int main(int argc, char* argv[]) {
#ifdef USE_DOUBLE
    std::cout << "Using FP64 double precision" << std::endl;
#else
    std::cout << "Using FP32 single precision" << std::endl;
#endif


    cxxopts::Options argumentOptions("Heterogeneous Conjugate Gradients", "CG Algorithm with CPU-GPU co-execution");

    argumentOptions.add_options()
        ("A,path_A", "path to .txt file containing symmetric positive definite matrix A",
         cxxopts::value<std::string>())
        ("b,path_b", "path to .txt file containing the right-hand side b", cxxopts::value<std::string>())
        ("o,output", "path to the output directory", cxxopts::value<std::string>())
        ("d,gp_input", "path to the input data for GP matrix generation", cxxopts::value<std::string>())
        ("gp_output", "path to the output data for GP matrix generation", cxxopts::value<std::string>())
        ("m,mode",
         "specifies the load balancing mode between CPU and GPU, has to be 'static', 'runtime', 'power' or 'util'",
         cxxopts::value<std::string>())
        ("z,matrix_bsz", "block size for the symmetric matrix storage", cxxopts::value<int>())
        // ("w,wg_size", "work-group size for matrix-vector operations", cxxopts::value<int>())
        ("v,wg_size_vec", "work-group size for vector-vector operations", cxxopts::value<int>())
        ("s,wg_size_sp", "work-group size for the final scalar product step on GPUs", cxxopts::value<int>())
        ("i,i_max", "maximum number of iterations", cxxopts::value<int>())
        ("e,eps", "epsilon value for the termination of the cg algorithm", cxxopts::value<double>())
        ("u,update_int", "interval in which CPU/GPU distribution will be rebalanced", cxxopts::value<int>())
        ("g,init_gpu_perc", "initial proportion of work assigned to gpu", cxxopts::value<double>())
        ("r,write_result", "write the result vector x to a .txt file", cxxopts::value<bool>())
        ("f,cpu_lb_factor", "factor that scales the CPU times for runtime load balancing", cxxopts::value<double>())
        ("t,block_update_th",
         "when block count change during re-balancing is equal or below this number, no re-balancing occurs",
         cxxopts::value<std::size_t>())
        ("size", "size of the matrix if a matrix should be generated from input data", cxxopts::value<std::size_t>());

    const auto arguments = argumentOptions.parse(argc, argv);

    bool generateMatrix = false;
    std::string path_A;
    std::string path_b;
    std::string path_gp_input;
    std::string path_gp_output;
    if (arguments.count("path_A") && arguments.count("path_b")) {
        path_A = arguments["path_A"].as<std::string>();
        path_b = arguments["path_b"].as<std::string>();
    } else if (arguments.count("gp_input") && arguments.count("gp_output")) {
        path_gp_input = arguments["gp_input"].as<std::string>();
        path_gp_output = arguments["gp_output"].as<std::string>();
        generateMatrix = true;
        if (arguments.count("size")) {
            conf::N = arguments["size"].as<std::size_t>();
        }
    } else {
        throw std::runtime_error(
            "No path to .txt file for matrix A specified and no path to input data for matrix generation specified");
    }

    if (arguments.count("output")) {
        conf::outputPath = arguments["output"].as<std::string>();
    }

    if (arguments.count("mode")) {
        conf::mode = arguments["mode"].as<std::string>();
    } else {
        conf::mode = "runtime";
    }

    if (arguments.count("matrix_bsz")) {
        conf::matrixBlockSize = arguments["matrix_bsz"].as<int>();
    }
    conf::workGroupSize = conf::matrixBlockSize;

    if (arguments.count("wg_size_vec")) {
        conf::workGroupSizeVector = arguments["wg_size_vec"].as<int>();
    }

    if (arguments.count("wg_size_sp")) {
        conf::workGroupSizeFinalScalarProduct = arguments["wg_size_sp"].as<int>();
    }

    if (arguments.count("i_max")) {
        conf::iMax = arguments["i_max"].as<int>();
    }

    if (arguments.count("eps")) {
        conf::epsilon = arguments["eps"].as<double>();
    }

    if (arguments.count("update_int")) {
        conf::updateInterval = arguments["update_int"].as<int>();
    }

    if (arguments.count("init_gpu_perc")) {
        conf::initialProportionGPU = arguments["init_gpu_perc"].as<double>();
    }

    if (arguments.count("write_result")) {
        conf::writeResult = arguments["write_result"].as<bool>();
    }

    if (arguments.count("cpu_lb_factor")) {
        conf::runtimeLBFactorCPU = arguments["cpu_lb_factor"].as<double>();
    }

    if (arguments.count("block_update_th")) {
        conf::blockUpdateThreshold = arguments["block_update_th"].as<std::size_t>();
    }


    sycl::property_list properties{sycl::property::queue::enable_profiling()};

    queue gpuQueue(gpu_selector_v, properties);
    queue cpuQueue(cpu_selector_v, properties);

    std::cout << "GPU: " << gpuQueue.get_device().get_info<info::device::name>() << std::endl;
    std::cout << "CPU: " << cpuQueue.get_device().get_info<info::device::name>() << std::endl;

    // measure CPU idle power draw in Watts
    UtilityFunctions::measureIdlePowerCPU();

    std::shared_ptr<LoadBalancer> loadBalancer;
    if (conf::mode == "static") {
        loadBalancer = std::make_shared<StaticLoadBalancer>(conf::updateInterval, conf::initialProportionGPU);
    } else if (conf::mode == "runtime") {
        loadBalancer = std::make_shared<RuntimeLoadBalancer>(conf::updateInterval, conf::initialProportionGPU);
    } else if (conf::mode == "util") {
        loadBalancer = std::make_shared<UtilizationLoadBalancer>(conf::updateInterval, conf::initialProportionGPU);
    } else if (conf::mode == "power") {
        loadBalancer = std::make_shared<PowerLoadBalancer>(conf::updateInterval, conf::initialProportionGPU);
    } else {
        throw std::runtime_error(
            "Invalid mode selected: '" + conf::mode + "' --> must be 'static', 'runtime', 'power' or 'util'");
    }


    // generate or parse Symmetric matrix
    RightHandSide b = generateMatrix
                          ? MatrixGenerator::parseRHS_GP(path_gp_output, cpuQueue)
                          : MatrixParser::parseRightHandSide(path_b, cpuQueue);

    SymmetricMatrix A = generateMatrix
                            ? MatrixGenerator::generateSPDMatrix(path_gp_input, cpuQueue)
                            : MatrixParser::parseSymmetricMatrix(path_A, cpuQueue);

    // MatrixParser::writeFullMatrix("./A_GP_128", A);
    // MatrixParser::writeBlockedMatrix("./A_GP_100_blocked", A);
    // CG algorithm(A, b, cpuQueue, gpuQueue, loadBalancer);
    // algorithm.solveHeterogeneous();

    conf::fp_type* A_gpu = malloc_device<conf::fp_type>(A.matrixData.size(), gpuQueue);
    for (int i = 0; i < 20; ++i) {
        gpuQueue.submit([&](handler& h) {
            h.memcpy(A_gpu, A.matrixData.data(), A.matrixData.size() * sizeof(conf::fp_type));
        }).wait();
        MatrixOperations::cholesky_GPU_optimized(gpuQueue, A_gpu, 0,0);
        // MatrixOperations::cholesky_GPU_optimized(cpuQueue, A.matrixData.data(), 0,0);
        gpuQueue.wait();

        sycl::event event = MatrixMatrixOperations::triangularSolve_optimizedGPU(gpuQueue, A_gpu, 0,0,1,A.blockCountXY -1);
        // sycl::event event = MatrixMatrixOperations::triangularSolve(cpuQueue, A.matrixData.data(), 0,0, 1);

        std::cout << static_cast<double>(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
            event.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e6 << std::endl;
    }


    gpuQueue.submit([&](handler& h) {
        h.memcpy(A.matrixData.data(), A_gpu, A.matrixData.size() * sizeof(conf::fp_type));
    }).wait();

    // MatrixParser::writeFullMatrix("./A_chol_test", A);


    return 0;
}
