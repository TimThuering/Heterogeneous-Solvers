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


def generateRightHandSide(N, path):
    print(r'Generating right-hand side b of size {}'.format(N))
    generator = np.random.default_rng(seed=1)
    b = generator.uniform(-1, 1, N)

    output = open(path + '/b_{}.txt'.format(N), "w")
    output.write('# {}\n'.format(N))

    output.write(np.array2string(b, max_line_width=sys.maxsize, precision=20, suppress_small=False, floatmode='fixed',
                                 separator=';')[1:-1] + "\n")


# Usage: First arguments: N, second argument: path to output directory
if __name__ == '__main__':
    N = int(sys.argv[1])
    if len(sys.argv) >= 3:
        path = str(sys.argv[2])
    else:
        path = '.'
    generateMatrix(N, path)
    generateRightHandSide(N, path)
