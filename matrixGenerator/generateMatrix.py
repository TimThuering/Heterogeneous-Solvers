import sys

import scipy.sparse.linalg
from sklearn.datasets import make_spd_matrix
import numpy as np
import scipy.sparse.linalg as sp

def generateMatrix(N, path):
    print(r'Generating SPD matrix A of size {}x{}'.format(N, N))
    seed = 123
    np.random.seed(seed)

    # matrix = make_spd_matrix(N, random_state=seed)
    print('-- generate random numbers...')
    matrix = np.random.randn(N,N) #TODO check if ok
    print('-- make symmetric...')
    matrix = np.dot(matrix, matrix.T)
    print('-- add diagonal values...')
    matrix += np.eye(N,N)

    # print("Start checking if matrix is symmetric positive definite...")
    #
    # try:
    #     np.linalg.cholesky(matrix)
    #     print('Matrix is symmetric positive definite.')
    # except:
    #     print('matrix is not symmetric positive definite')
    #     sys.exit(-1)

    print("-- Start writing matrix...")

    with open(f"{path}/A_{N}.txt", "w", buffering=1024 * 1024) as output:
        output.write('# {}\n'.format(N))
        for i in range(N):
            if i % 1000 == 0:
                print('\t {}'.format(i))
            numbers = matrix[i][0:i + 1]
            line = ';'.join(np.char.mod('%.17g', numbers)) + '\n'
            output.write(line)

    return matrix


def generateRightHandSide(N, path):
    print(r'Generating right-hand side b of size {}'.format(N))
    generator = np.random.default_rng(seed=1)
    b = generator.uniform(-1, 1, N)

    output = open(path + '/b_{}.txt'.format(N), "w")
    output.write('# {}\n'.format(N))

    output.write(np.array2string(b, max_line_width=sys.maxsize, precision=20, suppress_small=False, floatmode='fixed',
                                 separator=';', threshold=sys.maxsize)[1:-1] + "\n")

    return b


# Usage: First arguments: N, second argument: path to output directory
if __name__ == '__main__':
    N = int(sys.argv[1])
    if len(sys.argv) >= 3:
        path = str(sys.argv[2])
    else:
        path = './datasets/new/small'

    A = generateMatrix(N, path)
    b = generateRightHandSide(N, path)