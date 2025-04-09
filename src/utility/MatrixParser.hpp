#ifndef MATRIXPARSER_HPP
#define MATRIXPARSER_HPP

#include <string>

#include "SymmetricMatrix.hpp"
#include "Configuration.hpp"

/**
 * Class that contains functions to parse matrices from .txt files
 */
class MatrixParser
{
public:
    /**
     * Parses a symmetric matrix.
     * Stores the matrix in a blocked manner as described in the SymmetricMatrix class
     */
    static SymmetricMatrix parseSymmetricMatrix(std::string& path);

    static std::vector<conf::fp_type> parseRightHandSide(std::string& path);

    /**
     * Splits a string containing matrix entries.
     *
     * @return a vector containing those entries
     */
    static std::vector<conf::fp_type> getRowValuesFromString(const std::string& rowString);

    /**
     * Writes the symmetric matrix into a txt file for debugging purposes.
     * The diagonal blocks, where entries get mirrored will be represented too.
     */
    static void writeBlockedMatrix(const std::string& path, const SymmetricMatrix& matrix);

private:
    /**
     * Helper method used by parseSymmetricMatrix that processes a row of the matrix file and correctly stores all
     * values into the internal matrix data structure.
     */
    static void processRow(const std::string& row, unsigned int rowIndex, SymmetricMatrix& matrix);
};


#endif //MATRIXPARSER_HPP
