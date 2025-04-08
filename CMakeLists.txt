cmake_minimum_required(VERSION 3.9)
project(Heterogeneous_Conjugate_Gradients)

set(CMAKE_CXX_STANDARD 20)

option(ENABLE_TESTS "Enable Test" ON)
option(USE_DOUBLE "Use FP64 precision instead of FP32 for floating point data types" ON)




# find AdaptiveCpp SYCL implementation
find_package(AdaptiveCpp CONFIG REQUIRED)

# find OpenMP
find_package(OpenMP REQUIRED)

# include FetchContent
include(FetchContent)


# define all program source files for the executable
set(EXECUTABLE_FILES src/main.cpp)

# define all library files that will be linked to the executable
set(LIBRARY_FILES src/matrixData/SymmetricMatrix.cpp
        src/utility/Configuration.hpp
        src/utility/MatrixParser.cpp
        src/utility/MatrixParser.hpp)
add_library(Heterogeneous_CG_lib ${LIBRARY_FILES})

include_directories(src)
include_directories(src/matrixData)
include_directories(src/utility)

# create and executable, add SYCL to executable and link OpenMP and custom library files
add_executable(Heterogeneous_CG ${EXECUTABLE_FILES})
target_link_libraries(Heterogeneous_CG PUBLIC Heterogeneous_CG_lib)
add_sycl_to_target(TARGET Heterogeneous_CG SOURCES ${EXECUTABLE_FILES})
target_link_options(Heterogeneous_CG PUBLIC -fopenmp)

if (USE_DOUBLE)
    target_compile_definitions(Heterogeneous_CG_lib PUBLIC USE_DOUBLE=1)
endif ()

message("Building the hardware sampling library")
# build and link the hws library
set(HWS_ENABLE_PYTHON_BINDINGS OFF CACHE INTERNAL "" FORCE)
FetchContent_Declare(
        hws
        GIT_REPOSITORY https://github.com/SC-SGS/hardware_sampling
        GIT_TAG c78d3ec832a2b59259689ec7a3c96efda640f4d2
)
FetchContent_MakeAvailable(hws)
add_dependencies(Heterogeneous_CG hws)
target_include_directories(Heterogeneous_CG PUBLIC ${hws_SOURCE_DIR}/include)
target_link_libraries(Heterogeneous_CG PUBLIC hws::hws)
target_link_libraries(Heterogeneous_CG_lib PUBLIC hws::hws)

# Tests
if (${ENABLE_TESTS})
    set(TEST_SOURCE_FILES
            tests/testSymmetricMatrix.cpp)

    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.16.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(Heterogeneous_CG_Test ${TEST_SOURCE_FILES})
    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(Heterogeneous_CG_Test)
    target_link_libraries(Heterogeneous_CG_Test GTest::gtest_main Heterogeneous_CG_lib)
    message("Tests are enabled")
endif ()