import sys
from sklearn.datasets import make_spd_matrix
import numpy as np
import scipy.sparse.linalg as sp


def generateMatrix(N, path):
    print(r'Generating SPD matrix A of size {}x{}'.format(N, N))
    seed = 123

    matrix = make_spd_matrix(N, random_state=seed)
    matrix = np.array(matrix)

    output = open(path + '/A_{}.txt'.format(N), "w")
    output.write('# {}\n'.format(N))
    for i in range(N):
        output.write(np.array2string(matrix[i][0:i + 1], max_line_width=sys.maxsize, precision=20, suppress_small=False,
                                     floatmode='fixed', separator=';', threshold=sys.maxsize)[1:-1] + "\n")

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
        path = './datasets'

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
    # result = A[18:,0:18] @ b[0:18]
    # result = b + b
    # y = [np.sqrt(i) for i in range(N)]
    # result = b - y
    # result = [i ** 2 for i in b]
    # # print(sum(result[12:20]))
    # s = np.sum(A, axis=0)
    # print(sp.cg(A,b))
    # x = np.zeros(N)
    # r = b - A @ x
    # print(r.T @ r)
    result = b + 1.23456 *b
    print(np.array2string(result,precision=15, separator=','))

    # r = np.zeros(N)
    # d = np.zeros(N)
    # x = np.zeros(N)
    # q = np.zeros(N)
    #
    # r = b - A @ x
    # d = r
    # delta_new = r.T @ r
    # delta_zero = delta_new
    #
    # iteration = 0
    # while delta_new > (1 * 10 ** -6) ** 2 * delta_zero:
    #     q = A @ d
    #     alpha = delta_new / (d.T @ q)
    #     x = x + alpha * d
    #     if iteration % 50 == 0:
    #         r = b - A @ x
    #     else:
    #         r = r - alpha * q
    #
    #     delta_old = delta_new
    #     delta_new = r.T @ r
    #     if delta_new > delta_old:
    #         print(iteration, ": " ,delta_new , "<--" , delta_old)
    #     beta = delta_new / delta_old
    #     d = r + beta * d
    #     iteration += 1
    #
    # print(iteration)
