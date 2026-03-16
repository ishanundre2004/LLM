cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_HOST_COMPILER
"C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64")

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE CUDA_SOURCES
    tensor/*.cu
    kernels/**/*.cu
)

add_executable(main ${CUDA_SOURCES})

target_include_directories(main PRIVATE
    tensor
    kernels
)