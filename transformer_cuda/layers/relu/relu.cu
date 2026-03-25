#include<cuda_runtime.h>
#include "relu.h"

__global__ void relu_kernel(float* data, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size){
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

void relu(Tensor& t){
    int size = t.rows * t.cols;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(t.data, size);
    cudaDeviceSynchronize();
}