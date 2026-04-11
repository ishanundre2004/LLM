#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../../tensor/tensor.h"
#include "../../attention/qkv/qkv.h"  // Add this for matmul declaration

void fused_embedding_positional(
    Tensor& embedding,
    int* tokens,
    Tensor& output,
    int seq_len
);

void transformer_block(
    Tensor& input,
    Tensor& Wqkv,
    Tensor& W1, Tensor& b1,
    Tensor& W2, Tensor& b2,
    Tensor& output,
    int num_heads
);

void transformer_forward(
    int* tokens,
    Tensor& embedding,
    Tensor** Wqkv,
    Tensor** W1, Tensor** b1,
    Tensor** W2, Tensor** b2,
    Tensor& W_vocab,
    Tensor& output,
    int num_layers,
    int num_heads,
    int seq_len
);

#endif