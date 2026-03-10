#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<cuda_runtime.h>

struct Tensor
{
    float* data;
    int rows;
    int cols;

    Tensor(int r,int c);
    ~Tensor();

    void toGPU(float* host_data);
    void toCPU(float* host_data);

    void fill(float value);
    void print(int limit=10);
};

#endif;
