cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(SGEMM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Add the kernels directory to the include path
include_directories(${CMAKE_SOURCE_DIR}/kernels)

# List all source files
set(SOURCES
    main.cu
    kernels/00_cublas_sgemm.cu
    kernels/01_naive_sgemm.cu
    kernels/02_global_mem_coalesce.cu
    kernels/03_shared_mem_block.cu
    kernels/04_1D_block_tiling.cu
    kernels/05_2D_block_tiling.cu
    kernels/06_vectorized.cu
)

# Create the executable
add_executable(sgemm ${SOURCES})

target_link_libraries(sgemm -lcublas)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75) 
