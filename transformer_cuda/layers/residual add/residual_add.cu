#include<cuda_runtime.h>
#include "residual_add.h"

__global__ void residual_add_kernel(float* A, float* B, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size){
        A[idx] += B[idx];
    }
}

void residual_add(Tensor& input, Tensor& output){
    int size = input.size();
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    residual_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(input.data(), output.data(), size);
    cudaDeviceSynchronize();
}