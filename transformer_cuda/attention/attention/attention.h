#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

// Computes scaled dot-product attention:
//
// Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
//
// Q : (seq_len x d_k)
// K : (seq_len x d_k)
// V : (seq_len x d_k)
// O : (seq_len x d_k)
//
void attention(
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& O
);

#endif