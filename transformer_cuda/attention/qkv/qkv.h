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

void matmul(
    Tensor& A, 
    Tensor& B, 
    Tensor& C
);

void split_qkv(
    Tensor& QKV, 
    Tensor& Q, 
    Tensor& K, 
    Tensor& V
);


#endif