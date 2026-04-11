cmake_minimum_required(VERSION 3.18)

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Force MSVC compilers
set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64/cl.exe")
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64/cl.exe")

file(GLOB_RECURSE CUDA_SOURCES
    main.cu
    tensor/*.cu
    model/**/*.cu
    attention/**/*.cu
    layers/**/*.cu
    kernels/**/*.cu
)

add_executable(main ${CUDA_SOURCES})

target_include_directories(main PRIVATE
    .
    tensor
    model
    attention
    layers
    kernels
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(main CUDA::cublas CUDA::cudart)