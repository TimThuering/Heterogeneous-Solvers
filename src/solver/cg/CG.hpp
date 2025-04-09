#ifndef CG_HPP
#define CG_HPP

#include <string>
#include <vector>

#include "SymmetricMatrix.hpp"
#include "Configuration.hpp"

class CG
{
public:
    CG(std::string& path_A, std::string& path_b);
    SymmetricMatrix A;
    std::vector<conf::fp_type> b;

    void solve();
};


#endif //CG_HPP
