#ifndef MATRIXPARSER_HPP
#define MATRIXPARSER_HPP

#include <string>

#include "Configuration.hpp"
#include "SymmetricMatrix.hpp"
#include "RightHandSide.hpp"

/**
 * Class that contains functions to parse matrices from .txt files
 */
class MatrixParser {
public:
    /**
     * Parses a symmetric matrix.
     * Stores the matrix in a blocked manner as described in the SymmetricMatrix class
     *
     * @param path to the .txt file that stores the symmetric matrix
     * @param queue SYCL for allocating memory
     * @return the symmetric matrix object
     */
    static SymmetricMatrix parseSymmetricMatrix(std::string &path, sycl::queue &queue);

    /**
     * Parses data for the right-hand side
     *
     * @param path to the .txt file that stores the right-hand side
     * @param queue SYCL for allocating memory
     * @return the right-hand side object
     */
    static RightHandSide parseRightHandSide(std::string &path, sycl::queue &queue);

    /**
     * Splits a string containing matrix entries.
     *
     * @param rowString a string with floating point values seperated by ';'
     * @return a vector containing those entries
     */
    static std::vector<conf::fp_type> getRowValuesFromString(const std::string &rowString);


    /**
     * Writes the symmetric matrix into a txt file for debugging purposes.
     * The diagonal blocks, where entries get mirrored will be represented too.
     *
     * @param path output path
     * @param matrix the symmtetric matrix
     */
    static void writeBlockedMatrix(const std::string &path, const SymmetricMatrix &matrix);

private:

    /**
     * Helper method used by parseSymmetricMatrix that processes a row of the matrix file and correctly stores all
     * values into the internal matrix data structure.
     *
     * @param row string of the current matrix row to be processed
     * @param rowIndex index of the current row
     * @param matrix Symmetric matrix in which the row will be inserted
     */
    static void processRow(const std::string &row, unsigned int rowIndex, SymmetricMatrix &matrix);
};


#endif //MATRIXPARSER_HPP
