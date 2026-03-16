#include<cuda_runtime.h>
#include "embedding.h"

__global__ void embedding_kernel(
    float* embeddings, 
    int* tokens, 
    float* output,
    int seq_len,
    int d_model
){
    int idx = blockIdx.x + blockDim.x * threadIdx.x;

    int total  = seq_len * d_model;
    if(idx  > total) return;

    int token_pos = idx / d_model;
    int feature = idx % d_model;

    int token_id = tokens[token_pos];
    output[idx] = embeddings[token_id * d_model + feature];
}
