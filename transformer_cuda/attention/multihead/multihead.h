#pragma once
#include "../../tensor/tensor.h"


void split_heads(
    Tensor& input,
    Tensor& output,
    int seq_len,
    int d_model,
    int num_heads,
    int d_k
)

void concat_heads(
    Tensor& input,
    Tensor& output,
    int seq_len,
    int num_heads,
    int d_k
)

void multihead_attention(
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& output,
    int num_heads,
)

__global__ void split_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int d_model,
    int num_heads,
    int d_k
)
__global__ void concat_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int num_heads,
    int d_k
)

__global__ void mha_attention_kernel(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int d_k,
    int num_heads,
)