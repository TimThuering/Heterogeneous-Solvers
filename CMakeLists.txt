cmake_minimum_required(VERSION 3.28)
project(Heterogeneous_Conjugate_Gradients)

set(CMAKE_CXX_STANDARD 20)



find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(Heterogeneous_CG main.cpp)
add_sycl_to_target(TARGET Heterogeneous_CG SOURCES main.cpp)
target_link_options(Heterogeneous_CG PUBLIC -fopenmp)

