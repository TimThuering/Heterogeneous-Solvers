import sys
from sklearn.datasets import make_spd_matrix
import numpy as np


def generateMatrix(N, path):
    print(r'Generating SPD matrix A of size {}x{}'.format(N, N))
    seed = 123

    matrix = make_spd_matrix(N, random_state=seed)
    matrix = np.array(matrix)

    output = open(path + '/A_{}.txt'.format(N), "w")
    output.write('# {}\n'.format(N))
    for i in range(N):
        output.write(np.array2string(matrix[i][0:i + 1], max_line_width=sys.maxsize, precision=20, suppress_small=False,
                                     floatmode='fixed', separator=';')[1:-1] + "\n")

    return matrix


def generateRightHandSide(N, path):
    print(r'Generating right-hand side b of size {}'.format(N))
    generator = np.random.default_rng(seed=1)
    b = generator.uniform(-1, 1, N)

    output = open(path + '/b_{}.txt'.format(N), "w")
    output.write('# {}\n'.format(N))

    output.write(np.array2string(b, max_line_width=sys.maxsize, precision=20, suppress_small=False, floatmode='fixed',
                                 separator=';')[1:-1] + "\n")

    return b


# Usage: First arguments: N, second argument: path to output directory
if __name__ == '__main__':
    N = int(sys.argv[1])
    if len(sys.argv) >= 3:
        path = str(sys.argv[2])
    else:
        path = '.'
    A = generateMatrix(N, path)
    b = generateRightHandSide(N, path)

    # result = A[0:12,:] @ b
    # result = A[12:,:] @ b
    # result = A[0:12,0:12] @ b[0:12]
    # result = A[0:18,0:18] @ b[0:18]
    # result = A[12:,12:] @ b[12:]
    # result = A[18:,18:] @ b[18:]
    # result = A[0:12,12:] @ b[12:]
    # result = A[0:18,18:] @ b[18:]
    # result = A[12:,0:12] @ b[0:12]
    result = A[18:,0:18] @ b[0:18]
    s = np.sum(A, axis=0)
    print(np.array2string(result,precision=15, separator=','))
