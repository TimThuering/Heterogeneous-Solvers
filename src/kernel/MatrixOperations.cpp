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

    sycl::event event = queue.submit([&](sycl::handler& h) {
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

sycl::event MatrixOperations::cholesky_GPU(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow) {
    // launch kernel with the size of exactly one work-group
    const range globalRange(conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    const int blockStartIndex = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

    std::size_t N = conf::N;

    sycl::event event = queue.submit([&](sycl::handler& h) {
        auto local_column = local_accessor<conf::fp_type, 1>(conf::matrixBlockSize, h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(0);

            for (int k = 0; k < matrixBlockSize; ++k) {
                const conf::fp_type sqrtDiag = sycl::sqrt(A[blockStartIndex + k * matrixBlockSize + k]);

                conf::fp_type A_ik = 0.0;

                if ((blockRow * matrixBlockSize + local_i) < N) {
                    // update column below diagonal value
                    if (local_i > k) {
                        A_ik = A[blockStartIndex + local_i * matrixBlockSize + k] / sqrtDiag;
                        A[blockStartIndex + local_i * matrixBlockSize + k] = A_ik;
                        local_column[local_i] = A_ik; // store value of column in local memory
                    }
                }

                nd_item.barrier();

                // a_kk = sqrt(a_kk)
                if (local_i == k) {
                    A[blockStartIndex + local_i * matrixBlockSize + k] = sqrtDiag;
                }


                if ((blockRow * matrixBlockSize + local_i) < N) {
                    // process lower triangle right to the updated column
                    for (int j = k + 1; j < matrixBlockSize; ++j) {
                        if (local_i >= j) {
                            const conf::fp_type A_jk = local_column[j];
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


sycl::event MatrixOperations::cholesky_GPU_optimized(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow) {
    // launch kernel with the size of exactly one work-group
    const range globalRange{static_cast<unsigned long>(conf::matrixBlockSize),2};
    const range localRange{static_cast<unsigned long>(conf::matrixBlockSize),2};
    const auto kernelRange = nd_range{globalRange, localRange};

    const int matrixBlockSize = conf::matrixBlockSize;

    const int blockStartIndex = blockID * conf::matrixBlockSize * conf::matrixBlockSize;

    std::size_t N = conf::N;

    sycl::event event = queue.submit([&](sycl::handler& h) {
        auto local_column = local_accessor<conf::fp_type, 1>(conf::matrixBlockSize, h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int local_i = nd_item.get_local_id(1);
            // const int local_j = nd_item.get_local_id(1);
            const int local_id = nd_item.get_local_linear_id();


            // printf("%i,%i: %i\n", local_i, local_j, nd_item.get_local_linear_id());

            for (int k = 0; k < matrixBlockSize; ++k) {
                const conf::fp_type sqrtDiag = sycl::sqrt(A[blockStartIndex + k * matrixBlockSize + k]);

                conf::fp_type A_ik = 0.0;

                if ((blockRow * matrixBlockSize + local_i) < N) {
                    // update column below diagonal value
                    if (local_i > k && local_id < matrixBlockSize) {
                        A_ik = A[blockStartIndex + local_i * matrixBlockSize + k] / sqrtDiag;
                        A[blockStartIndex + local_i * matrixBlockSize + k] = A_ik;
                        local_column[local_i] = A_ik; // store value of column in local memory
                    }

                }

                nd_item.barrier();

                if (local_i > k && local_id >= matrixBlockSize) {
                    A[blockStartIndex + local_i * matrixBlockSize + k] = local_column[local_i];
                }

                // a_kk = sqrt(a_kk)
                if (local_i == k && local_id >= matrixBlockSize) {
                    A[blockStartIndex + local_i * matrixBlockSize + k] = sqrtDiag;
                }


                if ((blockRow * matrixBlockSize + local_i) < N) {
                    // process lower triangle right to the updated column
                    for (int j = k + 1; j < matrixBlockSize; ++j) {
                        if (local_i >= j && local_id < matrixBlockSize) {
                            const conf::fp_type A_jk = local_column[j];
                            A[blockStartIndex + local_i * matrixBlockSize + j] = A[blockStartIndex + local_i *
                                matrixBlockSize + j] - A_ik * A_jk;
                        } else if ((local_i < j && local_id >= matrixBlockSize)) {
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
