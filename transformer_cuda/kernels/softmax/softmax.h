#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../tensor/tensor.h"

void softmax(Tensor& input, Tensor& output, int rows, int cols);
void softmax(Tensor& input, Tensor& output);
void softmax_scaled(Tensor& input, Tensor& output, float scale);
void softmax_inplace(Tensor& input, float scale = 1.0f);

#endif // SOFTMAX_H
