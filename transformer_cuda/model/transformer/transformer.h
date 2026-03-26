#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../../tensor/tensor.h"

// Forward declarations (you already implemented these)
void fused_embedding_positional(
    Tensor& embedding,
    int* tokens,
    Tensor& output,
    int seq_len
);

void transformer_block(
    Tensor& input,
    Tensor& Wqkv,
    Tensor& W1,
    Tensor& W2,
    Tensor& output,
    int num_heads
);

void softmax(Tensor& input, Tensor& output);

// Main Transformer forward
void transformer_forward(
    int* tokens,
    Tensor& embedding,
    Tensor* Wqkv,
    Tensor* W1,
    Tensor* W2,
    Tensor& W_vocab,
    Tensor& output,
    int num_layers,
    int num_heads,
    int seq_len
);

// Argmax kernel launcher
void argmax(
    Tensor& logits,
    int* output_tokens
);

#endif