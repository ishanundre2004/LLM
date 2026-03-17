#include<cuda_runtime.h>
#include<math.h>
#include "positional_encoding.h"

__global__ void positional_encoding_kernel(
    float* data,
    int seq_len,
    int d_model,
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = seq_len * d_model;

    if(idx >= total) return;

    int pos = idx / d_model;
    int i = idx % d_model;
    
    float angle = pos/powf(10000.0f, (2.0f * (i/2)) / d_model);

    float pe = (i % 2 == 0) ? sinf(angle) : cosf(angle);

    data[idx] += pe;
}

void add_positional_encoding(Tensor& input){
    int seq_len = input.shape[0];
    int d_model = input.shape[1];

    int total = seq_len * d_model;

    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    positional_encoding_kernel<<<numBlocks, blockSize>>>(input.data, seq_len, d_model);

    cudaDeviceSynchronize();
    cudaFree();
}
