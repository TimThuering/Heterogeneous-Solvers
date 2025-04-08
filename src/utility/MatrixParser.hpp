#ifndef MATRIXPARSER_HPP
#define MATRIXPARSER_HPP

#include "SymmetricMatrix.hpp"
#include "Configuration.hpp"


class MatrixParser
{
public:
    SymmetricMatrix parseSymmetricMatrix(std::string& path);

    std::vector<conf::fp_type> getRowValuesFromString(std::string& rowString);

    void writeBlockedMatrix(std::string& path, SymmetricMatrix& matrix);
};


#endif //MATRIXPARSER_HPP
