cmake_minimum_required(VERSION 3.18)

# Project name and version
project(Kernels VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")  # Adjust based on your GPU architecture

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/avg-pool-1d.cuh
    src/avg-pool-1d-simple.cuh
)

# Test files
set(TEST_SOURCES
    test/test-avg-pool-1d.cu
    test/test-avg-pool-1d-simple.cu
)

# Create main project target
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} INTERFACE ${CUDA_LIBRARIES})

# Create test executables
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} ${PROJECT_NAME})
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    
    # Add test
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Set output directories
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Enable testing
enable_testing()

# Print configuration info
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
