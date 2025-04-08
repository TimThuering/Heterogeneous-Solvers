#ifndef MATRIXPARSER_HPP
#define MATRIXPARSER_HPP

#include "SymmetricMatrix.hpp"
#include "Configuration.hpp"


class MatrixParser
{
public:
    SymmetricMatrix parseSymmetricMatrix(std::string& path);

    std::vector<conf::fp_type> getRowValuesFromString(const std::string& rowString);

    void writeBlockedMatrix(const std::string& path, const SymmetricMatrix& matrix);

private:
    void processRow(std::string& row, unsigned int rowIndex, SymmetricMatrix& matrix);
};


#endif //MATRIXPARSER_HPP
