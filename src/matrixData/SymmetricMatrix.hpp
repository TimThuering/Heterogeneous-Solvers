#ifndef SYMMETRICMATRIX_HPP
#define SYMMETRICMATRIX_HPP

#include <vector>
#include "Configuration.hpp"

class SymmetricMatrix
{
public:
    SymmetricMatrix(std::size_t N, std::size_t blockSize);

    const std::size_t N; /// Size N of the NxN symmetric matrix
    const std::size_t blockSize; /// The matrix will be partitioned in blockSize x blockSize blocks
    const std::size_t blockCountXY; /// block Count in X/Y direction (if the matrix would be stored completely)

    std::vector<conf::fp_type> matrixData; /// internal matrix data structure

    int example();
};

#endif //SYMMETRICMATRIX_HPP
