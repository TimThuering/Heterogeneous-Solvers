#!/bin/bash

# This script installs the experiment environment with AdaptiveCpp
# AdaptiveCpp v25.02.0 is build against llvm 19.1.0
# Based on the official AdaptiveCpp documentation:
# https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-llvm.md
# https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md

set -e

if [ -z "$1" ]; then
  echo "Error: Missing argument for GPU VENDOR"
  exit 1
fi

GPU_VENDOR=${1} # can be "NVIDIA", "AMD" or "INTEL"


if [ -z "$2" ]; then
  echo "Error: Missing argument for BASE DIR"
  exit 1
fi

BASE_DIR=${2} # base directory in which everything will be installed


if [ -z "$3" ]; then
  echo "Error: Missing argument for CORE COUNT"
  exit 1
fi

CORE_COUNT=${3} # cores that are used to speed up compilation


if [ "${GPU_VENDOR}" == "AMD" ]; then
  if [ -z "$4" ]; then
    echo "Error: Missing argument ROCm PATH"
    exit 1
  fi
fi

if [ "${GPU_VENDOR}" == "AMD" ]; then
  ROCM_PATH=${4}
fi

mkdir -p ${BASE_DIR}

cd ${BASE_DIR} || exit


# install llvm 19
LLVM_INSTALL_PREFIX="${BASE_DIR}/llvm"
mkdir -p "${LLVM_INSTALL_PREFIX}"
cd ${LLVM_INSTALL_PREFIX} || exit
echo "installing llvm 19 into ${LLVM_INSTALL_PREFIX}"
echo "  -- cloning repository..."
git clone https://github.com/llvm/llvm-project -b llvmorg-19.1.0
cd llvm-project || exit
mkdir -p build
cd build || exit

echo "  -- building llvm 19..."
cmake -DCMAKE_C_COMPILER=`which gcc` -DCMAKE_CXX_COMPILER=`which g++` \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX}" \
      -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld;openmp" \
      -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86" \
      -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 \
      -DLLVM_INCLUDE_BENCHMARKS=0 \
      -DLLVM_INCLUDE_EXAMPLES=0 \
      -DLLVM_INCLUDE_TESTS=0 \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_INSTALL_RPATH="${LLVM_INSTALL_PREFIX}"/lib \
      -DLLVM_ENABLE_OCAMLDOC=OFF \
      -DLLVM_ENABLE_BINDINGS=OFF \
      -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
      -DLLVM_BUILD_LLVM_DYLIB=ON \
      -DLLVM_ENABLE_DUMP=OFF  ../llvm

make -j${CORE_COUNT} install

# Set llvm environment variables
export PATH=${LLVM_INSTALL_PREFIX}/bin:$PATH
export CMAKE_PREFIX_PATH=${LLVM_INSTALL_PREFIX}:$CMAKE_PREFIX_PATH
export CPATH=${LLVM_INSTALL_PREFIX}/include:$CPATH
export LIBRARY_PATH=${LLVM_INSTALL_PREFIX}/lib/x86_64-unknown-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=${LLVM_INSTALL_PREFIX}/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH


# install AdaptiveCpp
ACPP_INSTALL_PREFIX="${BASE_DIR}/acpp"
mkdir -p "${ACPP_INSTALL_PREFIX}"
cd "${ACPP_INSTALL_PREFIX}" || exit

echo "installing AdaptiveCpp v25.02.0 into ${ACPP_INSTALL_PREFIX}"
echo "  -- cloning repository..."
git clone https://github.com/AdaptiveCpp/AdaptiveCpp
cd AdaptiveCpp || exit
git checkout tags/v25.02.0
mkdir build
cd build || exit

echo "  -- building  AdaptiveCpp v25.02.0..."
if [ "${GPU_VENDOR}" == "NVIDIA" ]; then
  cmake -DCMAKE_INSTALL_PREFIX="${ACPP_INSTALL_PREFIX}" -DCMAKE_C_COMPILER="${LLVM_INSTALL_PREFIX}/bin/clang" -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_PREFIX}/bin/clang++" ..
elif [ "${GPU_VENDOR}" == "AMD" ]; then
  cmake -DCMAKE_INSTALL_PREFIX="${ACPP_INSTALL_PREFIX}" -DCMAKE_C_COMPILER="${LLVM_INSTALL_PREFIX}/bin/clang" -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_PREFIX}/bin/clang++" -DROCM_PATH="$ROCM_PATH" ..
elif [ "${GPU_VENDOR}" == "INTEL" ]; then
  cmake -DCMAKE_INSTALL_PREFIX="${ACPP_INSTALL_PREFIX}" -DCMAKE_C_COMPILER="${LLVM_INSTALL_PREFIX}/bin/clang" -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_PREFIX}/bin/clang++" -DWITH_OPENCL_BACKEND=ON ..
else
    echo "Unknown GPU vendor"
    exit 1
fi
make -j${CORE_COUNT} install



