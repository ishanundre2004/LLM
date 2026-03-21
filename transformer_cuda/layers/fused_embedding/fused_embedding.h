#ifndef FUSED_EMBEDDING_POSITIONAL_H
#define FUSED_EMBEDDING_POSITIONAL_H

#include "tensor.h"

// Applies token embedding lookup + sinusoidal positional encoding (fused)
// 
// embedding: [vocab_size x d_model]
// d_tokens:  device pointer to token indices [seq_len]
// output:    [seq_len x d_model]
// seq_len:   length of input sequence
//
void fused_embedding_positional(
    Tensor& embedding,
    int* d_tokens,
    Tensor& output,
    int seq_len
);

#endif