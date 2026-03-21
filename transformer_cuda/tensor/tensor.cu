#include "tensor.h"
#include<cuda_runtime.h>

__global__ void fill_kernel(float* data, float val, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        data[idx] = val;
    }
}



Tensor::Tensor(int r, int c){
    rows = r;
    cols = c;

    size_t size = rows * cols * sizeof(float);
    cudaMalloc(&data, size);
}

Tensor::~Tensor(){
    cudaFree(data);
}

void Tensor::toGPU(float* host_data){
    size_t size = rows * cols * sizeof(float);
    cudaMemcpy(data, host_data, size, cudaMemcpyHostToDevice );
}

void Tensor::toCPU(float* host_data){
    size_t size = rows * cols * sizeof(float);
    cudaMemcpy(host_data, data, size, cudaMemcpyDeviceToHost);
}

void Tensor::print(int limit){
    int size = rows * cols;
    float* host = new float[size];

    cudaMemcpy(host, data , size * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < std::min(size, limit); i++){
        std::cout << host[i] << " ";
    }
    std::cout << std::endl;
    delete[] host;
}

void Tensor::fill(float value){
    int size = rows * cols;

    int block = 256;
    int grid = (size + block - 1) / block;
    fill_kernel<<<grid, block>>>(data, value , size);

    cudaDeviceSynchronize();
}