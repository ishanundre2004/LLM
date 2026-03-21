#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

struct Tensor
{
    float* data;
    int rows;
    int cols;

    Tensor(int r,int c);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void toGPU(float* host_data);
    void toCPU(float* host_data);

    void fill(float value);
    void print(int limit=10) const;

    int size() const { return rows * cols; }
};

#endif