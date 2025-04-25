#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <cstdio>
#include <string>
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
        int workGroupSizeFinalScalarProduct = 512; /// Work-group size for the scalar product on GPUs
        std::size_t iMax = 1e5; /// maximum number of iterations
        conf::fp_type epsilon = 1.0e-6; /// epsilon value for the termination of the cg algorithm
        int updateInterval = 10;
        double initialProportionGPU = 0.5;
        std::string outputPath = "./output";
    };

    Configuration& get();

    inline int& matrixBlockSize = get().matrixBlockSize; /// Block size for storing the symmetric matrix in memory

    inline int& workGroupSize = get().workGroupSize;

    inline int& workGroupSizeVector = get().workGroupSizeVector;

    inline int& workGroupSizeFinalScalarProduct = get().workGroupSizeFinalScalarProduct;

    inline std::size_t& iMax = get().iMax;

    inline conf::fp_type& epsilon = get().epsilon;

    inline int& updateInterval = get().updateInterval;

    inline double& initialProportionGPU = get().initialProportionGPU;

    inline std::string& outputPath = get().outputPath;
}

#endif //CONFIGURATION_HPP
