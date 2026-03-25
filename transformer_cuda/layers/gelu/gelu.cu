#include <cuda_runtime.h>
#include <math.h>
#include "gelu.h"

__global__ void gelu_kernel(float* data, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size){
        float x = data[idx];
        data[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

void gelu(Tensor& t){
    int size = t.rows * t.cols;
    int threads = 256;
    int blocks  =(size + threads - 1)/threads;

    gelu_kernel<<<blocks, threads>>>(t.data, size);
    cudaDeviceSynchronize();
}