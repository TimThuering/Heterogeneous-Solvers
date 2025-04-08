#include "MatrixParser.hpp"

#include <assert.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>


SymmetricMatrix MatrixParser::parseSymmetricMatrix(std::string& path)
{
    std::ifstream matrixInputStream(path);

    std::string row;

    // read first line
    std::getline(matrixInputStream, row);

    if (!row.starts_with("#"))
    {
        throw std::invalid_argument("Invalid matrix format. First line has to be '# <N>!'");
    }

    // retrieve dimension N of the matrix form the input file
    std::size_t N = std::stoul(row.substr(2, row.size() - 2));

    // create symmetric matrix
    SymmetricMatrix matrix(N, conf::matrixBlockSize);

    // read file row by row
    unsigned int rowIndex = 0;
    while (std::getline(matrixInputStream, row))
    {
        // process current row string and store values in matrix data structure
        processRow(row, rowIndex, matrix);
        rowIndex++;
    }

    matrixInputStream.close();
    return matrix;
}

std::vector<conf::fp_type> MatrixParser::getRowValuesFromString(const std::string& rowString)
{
    std::vector<conf::fp_type> rowValues;

    int lastSplitIndex = 0;
    for (unsigned int i = 0; i < rowString.size(); ++i) // iterate through the string
    {
        if (constexpr char delimiter = ';'; rowString[i] == delimiter) // if the delimiter is reached, split the string
        {
            std::string valueString = rowString.substr(lastSplitIndex, i - lastSplitIndex);

            // cast the value to conf::fp_type and store the result
            try
            {
                auto value = static_cast<conf::fp_type>(std::stod(valueString));
                rowValues.push_back(value);
            } catch (...)
            {
                std::cerr << "Error while parsing string '" << valueString << "'" << std::endl;
            }
            lastSplitIndex = i + 1;
        }
    }

    // parse the last value
    const std::string valueString = rowString.substr(lastSplitIndex, rowString.size() - lastSplitIndex);
    const auto value = static_cast<conf::fp_type>(std::stod(valueString));
    rowValues.push_back(value);

    return rowValues;
}

void MatrixParser::processRow(std::string& row, const unsigned int rowIndex, SymmetricMatrix& matrix)
{
    const std::vector<conf::fp_type> rowValues = getRowValuesFromString(row);
    assert(rowValues.size() == rowIndex + 1);

    // Number of blocks in column direction in the current row
    const int columnBlockCount = std::ceil(
        static_cast<double>(rowValues.size()) / static_cast<double>(conf::matrixBlockSize));

    // row index divided by the block size to determine block index later
    auto rowDivBlock = std::div(rowIndex, matrix.blockSize);

    const int rowBlockIndex = rowDivBlock.quot;

    int blockCountLeftColumns = 0;
    for (int columnBlockIndex = 0; columnBlockIndex < columnBlockCount; ++columnBlockIndex)
    {
        const int blockIndex = blockCountLeftColumns + (rowBlockIndex - columnBlockIndex);

        // start index of block in matrix data structure
        const int blockStartIndex = blockIndex * conf::matrixBlockSize * conf::matrixBlockSize;

        // row index in block
        const int value_i = rowDivBlock.rem;

        if (rowBlockIndex == columnBlockIndex) // Diagonal Block, some values have to be mirrored
        {
            for (unsigned int j = columnBlockIndex * conf::matrixBlockSize; j < rowValues.size(); ++j)
            // for each column in block
            {
                const int value_j = j - columnBlockIndex * conf::matrixBlockSize;
                const conf::fp_type value = rowValues[j];

                // location as read in file in lower triangle (i,j)
                matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j] = value;
                // mirrored value in upper triangle (j,i)
                matrix.matrixData[blockStartIndex + value_j * conf::matrixBlockSize + value_i] = value;
            }
        }
        else // normal block
        {
            for (int j = columnBlockIndex * conf::matrixBlockSize; j < columnBlockIndex *
                 conf::matrixBlockSize +
                 conf::matrixBlockSize; ++j) // for each column in block
            {
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


void MatrixParser::writeBlockedMatrix(const std::string& path, const SymmetricMatrix& matrix)
{
    std::ofstream output(path);

    output << std::setprecision(10) << std::fixed;

    for (int rowIndex = 0; rowIndex < matrix.blockCountXY * matrix.blockSize; ++rowIndex) // for each row
    {
        if (rowIndex % conf::matrixBlockSize == 0)
        {
            output << std::endl;
        }
        // row index divided by the block size to determine block index later
        auto rowDivBlock = std::div(rowIndex, matrix.blockSize);
        const int rowBlockIndex = rowDivBlock.quot;

        int blockCountLeftColumns = 0;
        for (int columnBlockIndex = 0; columnBlockIndex <= rowBlockIndex; ++columnBlockIndex)
        {
            const int blockIndex = blockCountLeftColumns + (rowDivBlock.quot - columnBlockIndex);

            // start index of block in matrix data structure
            const int blockStartIndex = blockIndex * conf::matrixBlockSize * conf::matrixBlockSize;

            // row index in block
            const int value_i = rowDivBlock.rem;

            // for each column in block
            for (int value_j = 0; value_j < conf::matrixBlockSize; ++value_j)
            {
                conf::fp_type value = matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j];
                if (value >= 0)
                {
                    output << " " << value << ";";
                }
                else
                {
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
