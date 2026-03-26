#ifndef QKV_H
#define QKV_H

#include "../../tensor/tensor.h"

void compute_qkv(
    Tensor& X,
    Tensor& Wqkv,
    Tensor& Q,
    Tensor& K,
    Tensor& V
);

#endif