# Ensure nvcc accepts newer MSVC: add override before compiler detection
cmake_minimum_required(VERSION 3.18)

# Allow nvcc to override unsupported MSVC checks during detection
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(transformer_tests LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Force MSVC host compiler for CUDA
set(CMAKE_CXX_COMPILER
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64/cl.exe"
)

set(CMAKE_CUDA_HOST_COMPILER
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64/cl.exe"
)

# Use only the main test runner executable to avoid duplicate test definitions
set(TEST_SOURCES
    test_runner.cu
)

# Gather all implementation source files needed by the test runner
set(TEST_IMPL_SOURCES
    ../tensor/tensor.cu
    ../attention/qkv/qkv.cu
    ../attention/multihead/multihead.cu
    ../layers/fused_embedding/fused_embedding.cu
    ../layers/layernorm/layernorm.cu
    ../layers/feedforward/ffn.cu
    ../layers/residual_add/residual_add.cu
    ../model/transformer/transformer.cu
    ../model/transformer_block/transformer_block.cu
    ../kernels/gelu/gelu.cu
    ../kernels/softmax/softmax.cu
)

# Create test executable
add_executable(tests ${TEST_SOURCES} ${TEST_IMPL_SOURCES})

# Include directories for all headers
target_include_directories(tests PRIVATE
    ..
    ../tensor
    ../attention
    ../attention/qkv
    ../attention/multihead
    ../layers
    ../layers/fused_embedding
    ../layers/feedforward
    ../layers/layernorm
    ../layers/residual_add
    ../kernels
    ../kernels/gelu
    ../kernels/softmax   # Added softmax headers
    ../model
    ../model/transformer
    ../model/transformer_block
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Explicit Windows SDK link directories for kernel32.lib and standard system libs
link_directories(
    "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64"
    "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/ucrt/x64"
)

# Link CUDA libraries
target_link_libraries(tests PRIVATE
    CUDA::cublas
    CUDA::cudart
)

# Compile options for better performance and debugging
target_compile_options(tests PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O2
        -lineinfo
    >
)

# Set CUDA architecture (adjust sm_75 based on your GPU)
# sm_75: Turing (RTX 20/GTX 16 series)
# sm_86: Ampere (RTX 30 series)
# sm_89: Ada Lovelace (RTX 40 series)
set_target_properties(tests PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
)