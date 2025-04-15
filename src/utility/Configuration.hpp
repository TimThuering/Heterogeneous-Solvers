#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

/**
 * Namespace that contains everything regarding the configuration of the program execution
 */
namespace conf {
#ifdef USE_DOUBLE
    typedef double fp_type; /// use double precision for all floating point operations
#else
    typedef float fp_type; /// use single precision for all floating point operations
#endif

    inline int matrixBlockSize = 32; /// Block size for storing the symmetric matrix in memory

    inline int workGroupSize = 32;

    inline int workGroupSizeVector = 32;

    inline std::size_t iMax = 1e5;

    inline conf::fp_type epsilon = 1.0e-6;
}

#endif //CONFIGURATION_HPP
