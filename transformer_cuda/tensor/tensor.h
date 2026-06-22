#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

struct Tensor{
    float* data = nullptr;
    int ndim = 0;
    int shape[4] = {1, 1, 1, 1}; // Default shape for a 4D tensor
    int strides[4] = {1, 1, 1, 1}; // Default strides for a 4D tensor
    int total_size = 0;

    Tensor() = default; // Default constructor
    Tensor(int r,int c); //2D tensor constructor
    Tensor(int d0, int d1, int d2); //3D tensor constructor
    Tensor(int d0, int d1, int d2, int d3); //4D tensor constructor
    Tensor(const vector<int>& dims); //N-D tensor constructor
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // ---- Move semantics (for returning from functions) ----
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // ---- Data transfer ----
    void toGPU(const float* host_data);
    void toCPU(float* host_data) const;
    void fill(float value);
    void print(int limit = 10) const;

    // ---- Shape helpers ----
    int size() const { return total_size; }
    int rows() const { return shape[0]; }
    int cols() const { return shape[1]; }
    int dim(int axis) const { return shape[axis]; }

    // ---- View operations (no copy, just reinterpret shape) ----
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor view(int d0, int d1) const;
    Tensor view(int d0, int d1, int d2) const;
    Tensor view(int d0, int d1, int d2, int d3) const;

    // ---- Slice operations ----
    Tensor slice(int axis, int start, int end) const;   // extract range

    // ---- Transpose ----
    Tensor transpose(int axis1, int axis2) const;

    // ---- Create new tensor (allocates new memory) ----
    Tensor copy() const;
    Tensor reshape(const std::vector<int>& new_shape) const;
};

// ---- Standalone operations (modify or return new tensor) ----
Tensor add(const Tensor& a, const Tensor& b);
Tensor concat(const Tensor& a, const Tensor& b, int axis);
Tensor scale(const Tensor& a, float scalar);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor batched_matmul(const Tensor& a, const Tensor& b);

// In-place operations
void add_inplace(Tensor& a, const Tensor& b);
void scale_inplace(Tensor& a, float scalar);

// ---- CUDA kernel declarations ----
__global__ void fill_kernel(float* data, float val, int size);
__global__ void add_kernel(float* out, const float* a, const float* b, int size);
__global__ void scale_kernel(float* out, const float* a, float scalar, int size);
__global__ void copy_kernel(float* dst, const float* src, int size);
__global__ void concat_kernel(float* out, const float* a, const float* b,
                               int a_size, int b_size);

#endif
