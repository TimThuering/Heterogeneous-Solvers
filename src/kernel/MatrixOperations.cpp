#include "MatrixOperations.hpp"

#include <sycl/sycl.hpp>

using namespace sycl;

sycl::event MatrixOperations::cholesky(sycl::queue& queue, conf::fp_type* A, const int blockID, const int blockRow) {
    // launch kernel with the size of exactly one work-group
    const range globalRange(conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    const int blockStartIndex = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

    std::size_t N = conf::N;

    sycl::event event = queue.submit([&](handler& h) {
        h.parallel_for(kernelRange, [=](auto& nd_item) {
            int local_i = nd_item.get_local_id(0);

            for (int k = 0; k < matrixBlockSize; ++k) {
                const conf::fp_type sqrtDiag = sycl::sqrt(A[blockStartIndex + k * matrixBlockSize + k]);
                // a_kk = sqrt(a_kk)


                if ((blockRow * matrixBlockSize + local_i) < N) {
                    // update column below diagonal value
                    if (local_i > k) {
                        A[blockStartIndex + local_i * matrixBlockSize + k] = A[blockStartIndex + local_i *
                            matrixBlockSize +
                            k] / sqrtDiag;
                    }
                }

                nd_item.barrier();

                if (local_i == k) {
                    A[blockStartIndex + local_i * matrixBlockSize + k] = sqrtDiag;
                }


                if ((blockRow * matrixBlockSize + local_i) < N) {
                    // process lower triangle right to the updated column
                    for (int j = k + 1; j < matrixBlockSize; ++j) {
                        if (local_i >= j) {
                            const conf::fp_type A_ik = A[blockStartIndex + local_i * matrixBlockSize + k];
                            const conf::fp_type A_jk = A[blockStartIndex + j * matrixBlockSize + k];
                            A[blockStartIndex + local_i * matrixBlockSize + j] = A[blockStartIndex + local_i *
                                matrixBlockSize + j] - A_ik * A_jk;
                        } else {
                            A[blockStartIndex + local_i * matrixBlockSize + j] = 0;
                        }
                    }
                }

                nd_item.barrier();
            }
        });
    });

    return event;
}
