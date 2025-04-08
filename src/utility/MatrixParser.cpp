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

    SymmetricMatrix matrix(N, conf::matrixBlockSize);

    int rowIndex = 0;
    while (std::getline(matrixInputStream, row))
    {
        std::vector<conf::fp_type> rowValues = getRowValuesFromString(row);
        assert(rowValues.size() == rowIndex + 1);

        // Number of blocks in column direction in the current row
        int columnBlockCount = std::ceil(
            static_cast<double>(rowValues.size()) / static_cast<double>(conf::matrixBlockSize));

        // row index divided by the block size to determine block index later
        auto rowDivBlock = std::div(rowIndex, matrix.blockSize);

        int blockCountLeftColumns = 0;
        for (int columnBlockIndex = 0; columnBlockIndex < columnBlockCount; ++columnBlockIndex)
        {
            int blockIndex = blockCountLeftColumns + (rowDivBlock.quot - columnBlockIndex);

            // start index of block in matrix data structure
            int blockStartIndex = blockIndex * conf::matrixBlockSize * conf::matrixBlockSize;

            // row index in block
            int value_i = rowDivBlock.rem;

            if (rowDivBlock.quot == columnBlockIndex) // Diagonal Block, some values have to be mirrored
            {
                for (int value_j = columnBlockIndex * conf::matrixBlockSize; value_j < rowValues.size(); ++value_j)
                // for each column in block
                {
                    conf::fp_type value = rowValues[value_j];

                    // location as read in file in lower triangle (i,j)
                    matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j] = value;
                    // mirrored value in upper triangle (j,i)
                    matrix.matrixData[blockStartIndex + value_j * conf::matrixBlockSize + value_i] = value;
                }
            }
            else // normal block
            {
                for (int value_j = columnBlockIndex * conf::matrixBlockSize; value_j < columnBlockIndex *
                     conf::matrixBlockSize +
                     conf::matrixBlockSize; ++value_j) // for each column in block
                {
                    conf::fp_type value = rowValues[value_j];

                    // location as read in file in lower triangle (i,j)
                    matrix.matrixData[blockStartIndex + value_i * conf::matrixBlockSize + value_j] = value;
                }
            }

            // increment number of blocks in all columns left of the current column
            blockCountLeftColumns += matrix.blockCountXY - columnBlockIndex;
        }


        rowIndex++;
    }

    matrixInputStream.close();
    return matrix;
}

std::vector<conf::fp_type> MatrixParser::getRowValuesFromString(std::string& rowString)
{
    std::vector<conf::fp_type> rowValues;

    int lastSplitIndex = 0;
    for (auto i = 0; i < rowString.size(); ++i) // iterate through the string
    {
        if (constexpr char delimiter = ';'; rowString[i] == delimiter) // if the delimiter is reached, split the string
        {
            std::string valueString = rowString.substr(lastSplitIndex, i - lastSplitIndex);

            // cast the value to conf::fp_type and store the result
            conf::fp_type value = static_cast<conf::fp_type>(std::stod(valueString));
            rowValues.push_back(value);
            lastSplitIndex = i + 1;
        }
    }

    // parse the last value
    std::string valueString = rowString.substr(lastSplitIndex, rowString.size() - lastSplitIndex);
    conf::fp_type value = static_cast<conf::fp_type>(std::stod(valueString));
    rowValues.push_back(value);

    return rowValues;
}

void MatrixParser::writeBlockedMatrix(std::string& path, SymmetricMatrix& matrix)
{
    std::ofstream output(path);

    output << std::setprecision(10) << std::fixed;

    for (int rowIndex = 0; rowIndex < matrix.N; ++rowIndex) // for each row
    {
        if (rowIndex % conf::matrixBlockSize == 0)
        {
            output << std::endl;
        }
        // row index divided by the block size to determine block index later
        auto rowDivBlock = std::div(rowIndex, matrix.blockSize);
        int rowBlockIndex = rowDivBlock.quot;

        int blockCountLeftColumns = 0;
        for (int columnBlockIndex = 0; columnBlockIndex <= rowBlockIndex; ++columnBlockIndex)
        {
            int blockIndex = blockCountLeftColumns + (rowDivBlock.quot - columnBlockIndex);

            // start index of block in matrix data structure
            int blockStartIndex = blockIndex * conf::matrixBlockSize * conf::matrixBlockSize;

            // row index in block
            int value_i = rowDivBlock.rem;

            // for each column in block
            for (int value_j = columnBlockIndex * conf::matrixBlockSize; value_j < columnBlockIndex *
                 conf::matrixBlockSize +
                 conf::matrixBlockSize; ++value_j)
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
