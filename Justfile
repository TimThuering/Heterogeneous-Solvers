ACPP_PREFIX := "/opt/acpp"
ACPP_VERSION := "develop"
GPU_VENDOR := "AMD"
FP64 := "false"

build: _acpp_setup
  #! /usr/bin/env bash
  mkdir -p build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER={{ACPP_PREFIX}}/bin/acpp -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_PREFIX_PATH="{{ACPP_PREFIX}}" -DGPU_VENDOR={{GPU_VENDOR}} {{ if FP64 == "true" {"-DUSE_DOUBLE=ON"} else {"-DUSE_DOUBLE=OFF"} }} ..
  make -j $(nproc --all)

run *FLAGS:
  ./build/heterogeneous_solvers {{FLAGS}}

_acpp_setup:
  #! /usr/bin/env bash
  # Prematurely exit setup if installation can be found
  if [ -e {{ACPP_PREFIX}}/bin/acpp-info ]
  then exit
  fi
  cd ..
  git -C "AdaptiveCpp" pull origin {{ACPP_VERSION}} || git clone https://github.com/AdaptiveCpp/AdaptiveCpp
  cd AdaptiveCpp
  git checkout {{ACPP_VERSION}}
  mkdir -p build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX={{ACPP_PREFIX}} -DACPP_EXPERIMENTAL_LLVM=ON ..
  make -j $(nproc --all)
  make install
