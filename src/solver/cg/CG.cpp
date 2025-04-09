#include "CG.hpp"

#include <iostream>

#include "MatrixParser.hpp"

CG::CG(std::string& path_A, std::string& path_b): A(MatrixParser::parseSymmetricMatrix(path_A)),
                                                  b(MatrixParser::parseRightHandSide(path_b))
{
    // resize right hand side b to the size of the matrix A including padding
    if (A.N != b.size())
    {
        throw std::invalid_argument(
            "Dimensions of A and b do not match: " + std::to_string(A.N) + " != " + std::to_string(b.size()));
    }
    b.resize(A.blockSize * A.blockCountXY);
}

void CG::solve()
{
    std::cout << b.size() << std::endl;
    std::cout << b[0] << std::endl;
}
