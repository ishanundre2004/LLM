#pragma once
#include "../../tensor/tensor.h"

// Existing functions (unchanged)
void split_heads(
    Tensor& input,
    Tensor& output,
    int seq_len,
    int d_model,
    int num_heads,
    int d_k
);

void concat_heads(
    Tensor& input,
    Tensor& output,
    int seq_len,
    int num_heads,
    int d_k
);

void multihead_attention(
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& output,
    int num_heads
);

// New functions for masking support
void multihead_attention_causal(
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& output,
    int num_heads
);

void multihead_attention_masked(
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& mask,
    Tensor& output,
    int num_heads
);

// Existing kernels (unchanged)
__global__ void split_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int d_model,
    int num_heads,
    int d_k
);

__global__ void concat_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int num_heads,
    int d_k
);

__global__ void mha_attention_kernel(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int d_k,
    int num_heads
);

// New kernels for masking
__global__ void mha_attention_causal_kernel(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int d_k,
    int num_heads
);

__global__ void mha_attention_masked_kernel(
    float* Q,
    float* K,
    float* V,
    float* mask,
    float* O,
    int seq_len,
    int d_k,
    int num_heads
);