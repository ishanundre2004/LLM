#include <math.h>
#include <cuda_runtime.h>
#include "fused_embedding.h"

__global__ void fused_embedding_positional_kernel(
    float* embedding,
    int* tokens,
    float* output,
    int seq_len,
    int d_model
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;

    if (idx >= total) return;

    int pos = idx / d_model;
    int i   = idx % d_model;

    int token_id = tokens[pos];

    // STEP 1 — load embedding
    float val = embedding[token_id * d_model + i];

    // STEP 2 — compute positional encoding
    float angle = pos / powf(10000.0f, (2.0f * (i/2)) / d_model);

    float pe = (i % 2 == 0) ? sinf(angle) : cosf(angle);

    // STEP 3 — fused write
    output[idx] = val + pe;
}


void fused_embedding_positional(
    Tensor& embedding,
    int* d_tokens,
    Tensor& output,
    int seq_len
)
{
    int d_model = embedding.cols;
    int total = seq_len * d_model;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_embedding_positional_kernel<<<blocks, threads>>>(
        embedding.data,
        d_tokens,
        output.data,
        seq_len,
        d_model
    );

    cudaDeviceSynchronize();
}