#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <iostream>

namespace conf
{
#ifdef USE_DOUBLE
    typedef double fp_type;
#else
    typedef float fp_type;
#endif

    inline int matrixBlockSize = 2;
}

#endif //CONFIGURATION_HPP
