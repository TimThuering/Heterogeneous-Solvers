cmake_minimum_required(VERSION 3.9)
project(Heterogeneous_Conjugate_Gradients)

set(CMAKE_CXX_STANDARD 20)



find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(Heterogeneous_CG main.cpp)
add_sycl_to_target(TARGET Heterogeneous_CG SOURCES main.cpp)
target_link_options(Heterogeneous_CG PUBLIC -fopenmp)

include(FetchContent)
set(HWS_ENABLE_PYTHON_BINDINGS OFF CACHE INTERNAL "" FORCE)
FetchContent_Declare(
        hws
        GIT_REPOSITORY https://github.com/SC-SGS/hardware_sampling
        GIT_TAG v1.0.3
)
FetchContent_MakeAvailable(hws)
add_dependencies(Heterogeneous_CG hws)
target_include_directories(Heterogeneous_CG PUBLIC ${hws_SOURCE_DIR}/include)
target_link_libraries(Heterogeneous_CG PUBLIC hws::hws)

