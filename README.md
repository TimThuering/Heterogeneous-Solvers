# Heterogeneous Solvers for Linear Systems with SPD Matrices

This project was created as part of a master thesis.

It contains code for two heterogeneous solvers that can leverage the CPU and GPU simultaneously: a heterogeneous
implementation of the CG method and a heterogeneous Cholesky decomposition implementation.

The code is parallelized on the CPU and GPU using [SYCL](https://www.khronos.org/sycl/).

## Features

- Parallel, heterogeneous SYCL implementation of the CG method
- Parallel, heterogeneous SYCL implementation of the Cholesky decomposition
- GPU support for NVIDIA, AMD and Intel through SYCL
- Sampling of hardware metrics with the [hws-library](https://github.com/SC-SGS/hardware_sampling)

## Installation

The project supports the SYCL implementation [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) and requires a
Linux operating system.

The AdaptiveCpp compiler that has been used for the experiment environment can be installed using the script
`install_AdaptiveCpp.sh`.
The script installs AdaptiveCpp v25.02.0 and builds it against LLVM version 19.1.0 which is build from source.

Before running the script ensure that the CUDA/ROCm/oneAPI environment is loaded correctly. Usage:

```
./install_AdaptiveCpp.sh <GPU vendor: "NVIDIA", "AMD" or "INTEL"> <Base directory> <#Jobs for compilation (e.g. core count)> <AMD only: ROCm path>
```

Depending on the linux distribution and CUDA/ROCm/oneAPI setup, the script might not be able to install AdaptiveCpp automatically in every scenario.
Thus, a manual installation might still be required.

After the installation of AdaptiveCpp, clone this repository and create a build directory:

```
git clone https://github.com/TimThuering/Heterogeneous-Solvers.git
cd Heterogeneous-Solvers
mkdir build
cd build
```

### Building the project with AdaptiveCpp

The following command builds the Projects with the AdaptiveCpp CUDA backend for NVIDIA GPUs and the OpenMP backend for
CPUs.

Ensure that the CUDA environment and the AdaptiveCpp compiler are correctly loaded.
The project has been tested with CUDA 12.2.2 and
AdaptiveCpp [v25.02.0](https://github.com/AdaptiveCpp/AdaptiveCpp/tree/v25.02.0).

Replace `sm_XX` with the correct [compute capability](https://developer.nvidia.com/cuda-gpus) of your GPU, for example, `sm_80`.

```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=acpp -DACPP_TARGETS="cuda:sm_XX;omp.accelerated" -DCMAKE_CXX_FLAGS="-march=native" ..
make
```

Verify that the correct compilers are used by CMake. Alternatively specify the absolute path for `-DCMAKE_C_COMPILER`
and
`-DCMAKE_CXX_COMPILER`.
If problems occur during compilation try setting `-DAdaptiveCpp_DIR=<acpp install path>/lib/cmake/AdaptiveCpp`.
For further information about AdaptiveCpp please refer to the
official [documentation](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/using-acpp.md).

To build the project for AMD GPUs set the cmake variable `-DGPU_VENDOR="AMD"` and replace `cuda:sm_XX` with
`hip:gfxXXX`.
Set `gfxXXX` correctly according to your AMD GPU, for example, `gfx90a`.
Make sure that ROCm is loaded correctly before the installation. The project has been tested with ROCm 6.4.0.

To build the project for Intel GPUs set the cmake variable `-DGPU_VENDOR="INTEL"` and replace `cuda:sm_XX` with
`generic`.
Make sure that oneAPI is loaded correctly before the installation. The project has been tested with oneAPI 2025.1.

Building of tests can be enabled with the CMake option `-DENABLE_TESTS=true`.

### Running the program

The example below generates a kernel matrix, parses a right-hand side and solves the system heterogeneously on the GPU
and CPU with the Cholesky decomposition.

```
./heterogeneous_solvers --gp_input="<path to training input data>" --gp_output="<path to training output data>" --algorithm=cholesky --size=32768 --init_gpu_perc=0.45 --matrix_bsz=128
```

The tables below show a list of the mandatory and optional arguments to customize the program execution.

## Mandatory program arguments

| Argument          | Description                                 | Notes                                                                                                                                                |
|-------------------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--algorithm`     | the algorithm that should be used           | can be `cg` or `cholesky`                                                                                                                            | 
| `--init_gpu_perc` | initial proportion of work assigned to gpu  | always corresponds to the proportion of matrix block rows assigned to the GPU<br/> specifies the fixed GPU workload in case of static load balancing |
| `--matrix_bsz`    | block size for the symmetric matrix storage | must be a power of two, has to be >=64 for the most optimized GPU kernel for the Cholesky decomposition                                              |

### Mandatory program arguments to generate a kernel matrix

The first option to define the matrix for the linear system: Generation of a kernel matrix based on input data.
An exemplary dataset to generate such kernel matrices can, for example, be found in the repository
by [Helmann et al.](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4743).

| Argument      | Description                                                                | Notes             |
|---------------|----------------------------------------------------------------------------|-------------------|
| `--gp_input`  | path to the text file with (training) input data for GP matrix generation  | One entry per row |
| `--gp_output` | path to the text file with (training) output data for GP matrix generation | One entry per row |
| `--size`      | number that specifies the matrix side length of the kernel matrix          | -                 |

### Mandatory program arguments for the optional Gaussian Process Regression Pipeline

If `gpr` is not explicitly set to `true` no Gaussian Process Regression (GPR) is performed and only the linear system of
equations is solved.
If Gaussian Process Regression is desired, the following arguments are needed.

| Argument         | Description                                                  | Notes             |
|------------------|--------------------------------------------------------------|-------------------|
| `--gpr`          | perform gaussian process regression (GPR)                    | `true` or `false` |
| `--gp_test`      | path to the text file with (test) input data for GPR         | One entry per row |
| `--test_size`    | number that specifies the amount of test data read from file | -                 |
| `--write_result` | writes the result to a text file                             | `true` or `false` |

### Mandatory program arguments to use a custom linear system

The second option to define the matrix for the linear system: parse a file containing the matrix and parse a file
containing the right hand side.
Both files have to contain `# <N>` as the first line to specify the matrix side length `N`.
The file for the right-hand side has to store the vector in one row.
The entries in both files have to be separated with a semicolon followed by a space character.

| Argument   | Description                                                       | Notes             |
|------------|-------------------------------------------------------------------|-------------------|
| `--path_A` | path to .txt file containing symmetric positive-definite matrix A | One entry per row |
| `--path_b` | path to .txt file containing the right-hand side b                | One entry per row |

## Optional program arguments

| Argument                  | Description                                                                                                                                        | Notes                                                       |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `--mode`                  | specifies the load balancing mode between CPU and GPU, has to be `static`, `runtime` or `power`                                                    | Default: `static`                                           |
| `--output`                | path to the custom output directory                                                                                                                | -                                                           |
| `--i_max`                 | maximum number of iterations for the CG algorithm                                                                                                  | Default: 1e5                                                |
| `--eps`                   | epsilon value for the termination of the cg algorithm                                                                                              | Default: 1e-6                                               |
| `--update_int`            | interval in which CPU/GPU distribution will be rebalanced                                                                                          | Default: 10                                                 |
| `--write_result`          | write the result vector x to a .txt file                                                                                                           | Default: `false`                                            |
| `--write_matrix`          | write the result matrix L of the cholesky decomposition to a .txt file                                                                             | Default: `false`                                            |
| `--cpu_lb_factor`         | factor that scales the CPU times for runtime load balancing                                                                                        | Default: 1.2                                                |
| `--enableHWS`             | enables sampling with hws library, might affect CPU/GPU performance                                                                                | Default: `false`                                            |
| `--gpu_opt`               | optimization level 0-3 for GPU optimized matrix-matrix kernel (higher values for more optimized kernels), CG algorithm only supports 0 / greater 0 | Default: 3                                                  |
| `--cpu_opt`               | optimization level 0-2 for CPU optimized matrix-matrix kernel (higher values for more optimized kernels), CG algorithm only supports 0 / greater 0 | Default: 2; a value of 0 might be best for the CG algorithm |
| `--print_verbose`         | enable/disable verbose console output                                                                                                              | Default: `false`                                            |
| `--check_result`          | enable/disable result check that outputs error of Ax - b for the Cholesky decomposition                                                            | Default: `false`                                            |
| `--track_chol_solve`      | enable/disable hws tracking of solving step for the Cholesky decomposition                                                                         | Default: `true`                                             |
| `--unified_address_space` | assumes unified address space for CPU and GPU                                                                                                      | Default: `false`                                            |

It is recommended to specify the environment variables `OMP_NUM_THREADS` and `OMP_PROC_BIND` when using the CPU-only or
heterogeneous execution.
For the heterogeneous execution it is recommended to disable simultaneous multi threading.

Sampling of CPU metrics with the hws-library might require root privileges.

## References

The implementation of the heterogeneous CG algorithm is based on [Tiwari et al.](https://arxiv.org/pdf/2105.06176).
The implementation of the symmetric matrix-vector product is based on the approach
by [Nath et al.](https://dl.acm.org/doi/10.1145/2063384.2063392).
The GPU implementation of the scalar product is based on the method
by [Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).

The GPU implementations of the matrix-matrix multiplication kernels for the Cholesky decomposition are based
on [Rauber et al.](https://link.springer.com/book/10.1007/978-3-031-28924-8)
and [Tan et al.](https://dl.acm.org/doi/10.1145/2063384.2063431).