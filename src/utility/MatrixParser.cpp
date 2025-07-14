#include "MatrixParser.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>


SymmetricMatrix MatrixParser::parseSymmetricMatrix(std::string& path, sycl::queue& queue) {
    std::ifstream matrixInputStream(path);

    std::string row;

    // read first line
    std::getline(matrixInputStream, row);

    if (row[0] != '#' || row.empty()) {
        throw std::invalid_argument("Invalid matrix format. First line has to be '# <N>!'");
    }

    // retrieve dimension N of the matrix form the input file
    std::size_t N = std::stoul(row.substr(2, row.size() - 2));

    std::cout << "-- Starting to parse symmetric matrix of size " << N << "x" << N << std::endl;
    conf::N = N;
    // create symmetric matrix
    SymmetricMatrix matrix(N, conf::matrixBlockSize, queue);

    // read file row by row
    unsigned int rowIndex = 0;
    while (std::getline(matrixInputStream, row)) {
        // process current row string and store values in matrix data structure
        processRow(row, rowIndex, matrix);
        rowIndex++;
    }

    matrixInputStream.close();
    return matrix;
}

RightHandSide MatrixParser::parseRightHandSide(std::string& path, sycl::queue& queue) {
    std::ifstream rhsInputStream(path);

    std::string row;

    // read first line
    std::getline(rhsInputStream, row);

    if (row[0] != '#' || row.empty()) {
        throw std::invalid_argument("Invalid right-hand side format. First line has to be '# <N>!'");
    }

    // retrieve dimension N of the matrix form the input file
    const std::size_t N = std::stoul(row.substr(2, row.size() - 2));

    std::cout << "-- Starting to parse right-hand side of size " << N << std::endl;

    std::getline(rhsInputStream, row);

    auto values = getRowValuesFromString(row);

    RightHandSide b(N, conf::matrixBlockSize, queue);

    // copy values to the right-hand side vector b which is allocated as sycl host memory
    for (unsigned int i = 0; i < N; i++) {
        b.rightHandSideData[i] = values[i];
    }

    return b;
}

std::vector<conf::fp_type> MatrixParser::getRowValuesFromString(const std::string& rowString) {
    std::vector<conf::fp_type> rowValues;

    int lastSplitIndex = 0;
    for (unsigned int i = 0; i < rowString.size(); ++i) // iterate through the string
    {
        if (constexpr char delimiter = ';'; rowString[i] == delimiter) // if the delimiter is reached, split the string
        {
            std::string valueString = rowString.substr(lastSplitIndex, i - lastSplitIndex);

            // cast the value to conf::fp_type and store the result
            try {
                auto value = static_cast<conf::fp_type>(std::stod(valueString));
                rowValues.push_back(value);
            } catch (...) {
                throw std::invalid_argument("Error while parsing string '" + valueString + "'");
            }
            lastSplitIndex = i + 1;
        }
    }

    // parse the last value
    const std::string valueString = rowString.substr(lastSplitIndex, rowString.size() - lastSplitIndex);
    try {
        const auto value = static_cast<conf::fp_type>(std::stod(valueString));
        rowValues.push_back(value);
    } catch (...) {
        throw std::invalid_argument("Error while parsing string '" + valueString + "'");
    }

    return rowValues;
}

void MatrixParser::processRow(const std::string& row, const unsigned int rowIndex, SymmetricMatrix& matrix) {
    const std::vector<conf::fp_type> rowValues = getRowValuesFromString(row);
    assert(rowValues.size() == rowIndex + 1);

    // Number of blocks in column direction in the current row
    const int columnBlockCount = std::ceil(
        static_cast<double>(rowValues.size()) / static_cast<double>(conf::matrixBlockSize));

    // row index divided by the block size to determine block index later
    auto rowDivBlock = std::div(rowIndex, matrix.blockSize);

    // row index of the current block
    const int rowBlockIndex = rowDivBlock.quot;

    // tracks the total number of blocks in columns left to the current column
    int blockCountLeftColumns = 0;

    // iterate over all column indices of blocks that intersect with the current row
    for (int columnBlockIndex = 0; columnBlockIndex < columnBlockCount; ++columnBlockIndex) {
        // Index of block when enumerating all blocks column by column from top to bottom
        const int blockIndex = blockCountLeftColumns + (rowBlockIndex - columnBlockIndex);

        // start index of block in matrix data structure
        const std::size_t blockStartIndex = static_cast<std::size_t>(blockIndex) * conf::matrixBlockSize * conf::matrixBlockSize;

        // local row index in block
        const int value_i = rowDivBlock.rem;

        if (rowBlockIndex == columnBlockIndex) // Diagonal Block, some values have to be mirrored
        {
            // for each column in block, j is a global index and corresponds to the column of the whole matrix
            for (unsigned int j = columnBlockIndex * conf::matrixBlockSize; j < rowValues.size(); ++j) {
                // compute local column index value_j inside the current block from global column index j
                const unsigned int value_j = j - columnBlockIndex * conf::matrixBlockSize;

                const conf::fp_type value = rowValues[j];

                // location as read in file in lower triangle (i,j)
                matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j] = value;
                // mirrored value in upper triangle (j,i)
                matrix.matrixData[blockStartIndex + value_j * conf::matrixBlockSize + value_i] = value;
            }
        } else // normal block
        {
            // for each column in block, j is a global index and corresponds to the column of the whole matrix
            for (int j = columnBlockIndex * conf::matrixBlockSize; j < columnBlockIndex * conf::matrixBlockSize + conf::matrixBlockSize; ++j) // for each column in block
            {
                // compute local column index value_j inside the current block from global column index j
                const int value_j = j - columnBlockIndex * conf::matrixBlockSize;

                const conf::fp_type value = rowValues[j];

                // location as read in file in lower triangle (i,j)
                matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j] = value;
            }
        }

        // increment number of blocks in all columns left of the current column
        blockCountLeftColumns += matrix.blockCountXY - columnBlockIndex;
    }
}


void MatrixParser::writeBlockedMatrix(const std::string& path, const SymmetricMatrix& matrix) {
    std::ofstream output(path);

    output << std::setprecision(10) << std::fixed;

    for (int rowIndex = 0; rowIndex < matrix.blockCountXY * matrix.blockSize; ++rowIndex) // for each row
    {
        if (rowIndex % conf::matrixBlockSize == 0) {
            output << std::endl;
        }
        // row index divided by the block size to determine block index later
        auto rowDivBlock = std::div(rowIndex, matrix.blockSize);
        const int rowBlockIndex = rowDivBlock.quot;

        int blockCountLeftColumns = 0;
        for (int columnBlockIndex = 0; columnBlockIndex <= rowBlockIndex; ++columnBlockIndex) {
            const int blockIndex = blockCountLeftColumns + (rowDivBlock.quot - columnBlockIndex);

            // start index of block in matrix data structure
            const std::size_t blockStartIndex = static_cast<std::size_t>(blockIndex) * conf::matrixBlockSize * conf::matrixBlockSize;

            // row index in block
            const int value_i = rowDivBlock.rem;

            // for each column in block
            for (int value_j = 0; value_j < conf::matrixBlockSize; ++value_j) {
                conf::fp_type value = matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j];
                if (value >= 0) {
                    output << " " << value << ";";
                } else {
                    output << value << ";";
                }
            }

            // increment number of blocks in all columns left of the current column
            blockCountLeftColumns += matrix.blockCountXY - columnBlockIndex;

            output << "\t";
        }
        output << std::endl;
    }
    output.close();
}

void MatrixParser::writeFullMatrix(const std::string& path, const SymmetricMatrix& matrix) {
    std::ofstream output(path);

    output << std::setprecision(20) << std::fixed;

    for (int rowIndex = 0; rowIndex < matrix.blockCountXY * matrix.blockSize; ++rowIndex) // for each row
    {
        // row index divided by the block size to determine block index later
        auto rowDivBlock = std::div(rowIndex, matrix.blockSize);
        const int rowBlockIndex = rowDivBlock.quot;

        int blockIndex;
        int blockCountLeftColumns = 0;
        for (int columnBlockIndex = 0; columnBlockIndex <= rowBlockIndex; ++columnBlockIndex) {
            blockIndex = blockCountLeftColumns + (rowDivBlock.quot - columnBlockIndex);

            // start index of block in matrix data structure
            const std::size_t blockStartIndex = static_cast<std::size_t>(blockIndex) * conf::matrixBlockSize * conf::matrixBlockSize;

            // row index in block
            const int value_i = rowDivBlock.rem;

            // for each column in block
            for (int value_j = 0; value_j < conf::matrixBlockSize; ++value_j) {
                conf::fp_type value = matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j];
                if (value >= 0) {
                    output << " " << value << ";";
                } else {
                    output << value << ";";
                }
            }

            // increment number of blocks in all columns left of the current column
            blockCountLeftColumns += matrix.blockCountXY - columnBlockIndex;
        }

        int diagBlockIndex = blockIndex;
        for (int i = 0; i < matrix.blockCountXY - (rowBlockIndex + 1); ++i) {
            blockIndex = diagBlockIndex + i + 1;

            // start index of block in matrix data structure
            const std::size_t blockStartIndex = static_cast<std::size_t>(blockIndex) * conf::matrixBlockSize * conf::matrixBlockSize;

            // row index in block
            const int value_i = rowDivBlock.rem;

            // for each column in block
            for (int value_j = 0; value_j < conf::matrixBlockSize; ++value_j) {
                conf::fp_type value = matrix.matrixData[blockStartIndex + value_j * conf::matrixBlockSize + value_i];
                if (value >= 0) {
                    output << " " << value << ";";
                } else {
                    output << value << ";";
                }
            }
        }

        output << std::endl;
    }
    output.close();
}
