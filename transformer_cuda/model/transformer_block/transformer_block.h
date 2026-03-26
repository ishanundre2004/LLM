#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "../../tensor/tensor.h"

void transformer_block(
    Tensor& X,
    Tensor& Wqkv,
    Tensor& W1,
    Tensor& W2,
    Tensor& output,
    int num_heads
);

#endif
