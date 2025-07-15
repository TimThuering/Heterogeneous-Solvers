#include "MatrixOperations.hpp"

#include <sycl/sycl.hpp>

using namespace sycl;

sycl::event MatrixOperations::cholesky(sycl::queue& queue, conf::fp_type* A, const int blockID, const int blockRow) {
    // launch kernel with the size of exactly one work-group
    const range globalRange(conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const std::size_t matrixBlockSize = conf::matrixBlockSize;

    const std::size_t blockStartIndex = static_cast<std::size_t>(blockID) * conf::matrixBlockSize * conf::matrixBlockSize;

    const long N = static_cast<long>(conf::N);

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
                            matrixBlockSize + k] / sqrtDiag;
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
                            A[blockStartIndex + local_i * matrixBlockSize + j] = A[blockStartIndex + local_i * matrixBlockSize + j] - A_ik * A_jk;
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

    const std::size_t matrixBlockSize = conf::matrixBlockSize;

    const std::size_t blockStartIndex = static_cast<std::size_t>(blockID) * conf::matrixBlockSize * conf::matrixBlockSize;

    const long N = static_cast<long>(conf::N);

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
                            A[blockStartIndex + local_i * matrixBlockSize + j] = A[blockStartIndex + local_i * matrixBlockSize + j] - A_ik * A_jk;
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


sycl::event MatrixOperations::cholesky_optimizedGPU(sycl::queue& queue, conf::fp_type* A, int blockID, int blockRow) {
    // launch kernel with the size of exactly one work-group
    const range globalRange(conf::matrixBlockSize * conf::matrixBlockSize);
    const range localRange(conf::matrixBlockSize);
    const auto kernelRange = nd_range{globalRange, localRange};

    const std::size_t matrixBlockSize = conf::matrixBlockSize;

    const std::size_t blockStartIndex = static_cast<std::size_t>(blockID) * conf::matrixBlockSize * conf::matrixBlockSize;

    const long N = static_cast<long>(conf::N);

    sycl::event event = queue.submit([&](sycl::handler& h) {
        auto local_column = local_accessor<conf::fp_type, 1>(conf::matrixBlockSize, h);

        h.parallel_for(kernelRange, [=](auto& nd_item) {
            const int group_id = nd_item.get_group().get_group_id();
            const int local_i = nd_item.get_local_id(0);
            if (group_id == 0) {
                for (int k = 0; k < matrixBlockSize; ++k) {
                    conf::fp_type A_ik = 0.0;

                    if ((blockRow * matrixBlockSize + local_i) < N) {
                        // update column below diagonal value
                        if (local_i > k) {
                            const conf::fp_type sqrtDiag = sycl::sqrt(A[blockStartIndex + k * matrixBlockSize + k]);
                            A_ik = A[blockStartIndex + local_i * matrixBlockSize + k] / sqrtDiag;
                            A[blockStartIndex + local_i * matrixBlockSize + k] = A_ik;
                            local_column[local_i] = A_ik; // store value of column in local memory
                        }
                    }

                    nd_item.barrier();

                    if ((blockRow * matrixBlockSize + local_i) < N) {
                        // process lower triangle right to the updated column
                        for (int j = k + 1; j < matrixBlockSize; ++j) {
                            if (local_i >= j) {
                                const conf::fp_type A_jk = local_column[j];
                                A[blockStartIndex + local_i * matrixBlockSize + j] = A[blockStartIndex + local_i *
                                    matrixBlockSize + j] - A_ik * A_jk;
                            }
                        }
                    }

                    // a_kk = sqrt(a_kk)
                    if (local_i == 0) {
                        const conf::fp_type sqrtDiag = sycl::sqrt(A[blockStartIndex + k * matrixBlockSize + k]);
                        A[blockStartIndex + k * matrixBlockSize + k] = sqrtDiag;
                    }

                    nd_item.barrier();
                }
            } else {
                // set upper triangle to 0
                if ((blockRow * matrixBlockSize + local_i) < N) {
                    for (int j = group_id; j < matrixBlockSize; ++j) {
                        if (local_i < j) {
                            A[blockStartIndex + local_i * matrixBlockSize + j] = 0;
                        }
                    }
                }
            }
        });
    });

    return event;
}
