#ifndef CG_HPP
#define CG_HPP

#include <string>
#include <sycl/sycl.hpp>

#include "SymmetricMatrix.hpp"
#include "RightHandSide.hpp"
#include "Configuration.hpp"

using namespace sycl;

class CG {
public:
    CG(std::string &path_A, std::string &path_b, queue &cpuQueue, queue &gpuQueue);

    SymmetricMatrix A;
    RightHandSide b;

    queue &cpuQueue;
    queue &gpuQueue;


    void solve();
};


#endif //CG_HPP
