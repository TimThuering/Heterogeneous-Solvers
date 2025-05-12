#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <cstdio>
#include <string>
#include "hws/sample_category.hpp"

/**
 * Namespace that contains everything regarding the configuration of the program execution
 */
namespace conf {
#ifdef USE_DOUBLE
    typedef double fp_type; /// use double precision for all floating point operations
#else
    typedef float fp_type; /// use single precision for all floating point operations
#endif

    struct Configuration {
        int matrixBlockSize = 128; /// Block size for storing the symmetric matrix in memory
        int workGroupSize = 128; /// Work-group size for matrix-vector operations
        int workGroupSizeVector = 128; /// Work-group size for vector-vector operations
        int workGroupSizeFinalScalarProduct = 512; /// Work-group size for the final scalar product step on GPUs
        std::size_t iMax = 1e5; /// maximum number of iterations
        double epsilon = 1.0e-6; /// epsilon value for the termination of the cg algorithm
        int updateInterval = 10; /// interval in which CPU/GPU distribution will be rebalanced
        double initialProportionGPU = 0.5; /// initial proportion of work assigned to gpu
        std::string outputPath = "./output"; /// path for all output files
        bool writeResult = false; /// write the result vector into a .txt file
        std::string mode = "runtime"; /// mode for the heterogeneous scheduling
        std::size_t N = 0; /// size of the NxN matrix
        double runtimeLBFactorCPU = 1.2; /// factor that scales the CPU runtimes to influence the scheduling
        std::size_t blockUpdateThreshold = 1;
        /// when block count change during re-balancing is equal or below this number, no re-balancing occurs
        double idleWatt_CPU = 30.0; /// CPU power draw in Watts when the CPU is idle. Used to estimate total power draw.
        hws::sample_category sampleCategories = static_cast<hws::sample_category>(0b00000101); /// enable power and general samples
    };

    Configuration& get();

    inline int& matrixBlockSize = get().matrixBlockSize;

    inline int& workGroupSize = get().workGroupSize;

    inline int& workGroupSizeVector = get().workGroupSizeVector;

    inline int& workGroupSizeFinalScalarProduct = get().workGroupSizeFinalScalarProduct;

    inline std::size_t& iMax = get().iMax;

    inline double& epsilon = get().epsilon;

    inline int& updateInterval = get().updateInterval;

    inline double& initialProportionGPU = get().initialProportionGPU;

    inline std::string& outputPath = get().outputPath;

    inline bool& writeResult = get().writeResult;

    inline std::string& mode = get().mode;

    inline std::size_t& N = get().N;

    inline double& runtimeLBFactorCPU = get().runtimeLBFactorCPU;

    inline std::size_t& blockUpdateThreshold = get().blockUpdateThreshold;

    inline double& idleWatt_CPU = get().idleWatt_CPU;

    inline hws::sample_category& sampleCategories = get().sampleCategories;
}

#endif //CONFIGURATION_HPP
