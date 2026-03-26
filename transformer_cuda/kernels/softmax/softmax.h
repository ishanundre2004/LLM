#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../tensor/tensor.h"

// For explicit size controls (useful when input may alias output or for partial views)
void softmax(Tensor& input, Tensor& output, int rows, int cols);

// Convenience wrapper for full tensor softmax
void softmax(Tensor& input, Tensor& output);

#endif