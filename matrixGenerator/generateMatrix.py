import sys
from sklearn.datasets import make_spd_matrix
import numpy as np

def generateMatrix(N, path):

    print(r'Generating SPD matrix of size {}x{}'.format(N,N))
    seed = 123

    matrix = make_spd_matrix(N, random_state=seed)
    matrix = np.array(matrix)

    output = open(path + '/matrix.txt', "w")
    for i in range(N):
        output.write(np.array2string(matrix[i][0:i+1], max_line_width=sys.maxsize, precision=20, suppress_small=False, floatmode='fixed')[1:-1] + "\n")

# Usage: First arguments: N, second argument: path to output directory
if __name__ == '__main__':
    N = int(sys.argv[1])
    if len(sys.argv) >= 3:
        path = str(sys.argv[2])
    else:
        path = '.'
    generateMatrix(N, path)
