#include <cuda_runtime.h>
#include "residual_add.h"

//////////////////////////////////////////////////////////////
// KERNEL
//////////////////////////////////////////////////////////////

__global__ void residual_add_kernel(float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        A[idx] += B[idx];
    }
}

//////////////////////////////////////////////////////////////
// HOST FUNCTION
//////////////////////////////////////////////////////////////

void residual_add(Tensor& input, Tensor& output)
{
    int size = input.rows * input.cols;  // safer than size()

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    residual_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data,   // ✅ FIXED
        output.data,  // ✅ FIXED
        size
    );

    cudaDeviceSynchronize();
}