#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

// ---- Utility ----
static void compute_strides(int ndim, const int* shape, int* strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

static int compute_size(int ndim, const int* shape) {
    int s = 1;
    for (int i = 0; i < ndim; i++) s *= shape[i];
    return s;
}

// ---- Kernels ----
__global__ void fill_kernel(float* data, float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = val;
}

__global__ void add_kernel(float* out, const float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

__global__ void scale_kernel(float* out, const float* a, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] * scalar;
}

__global__ void copy_kernel(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] = src[idx];
}

__global__ void concat_kernel(float* out, const float* a, const float* b,
                               int a_size, int b_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = a_size + b_size;
    if (idx < total) {
        out[idx] = (idx < a_size) ? a[idx] : b[idx - a_size];
    }
}

static int get_grid(int size, int block = 256) {
    return (size + block - 1) / block;
}

// ---- Constructors ----
Tensor::Tensor(int r, int c) {
    ndim = 2;
    shape[0] = r; shape[1] = c;
    compute_strides(ndim, shape, strides);
    total_size = compute_size(ndim, shape);
    cudaMalloc(&data, total_size * sizeof(float));
}

Tensor::Tensor(int d0, int d1, int d2) {
    ndim = 3;
    shape[0] = d0; shape[1] = d1; shape[2] = d2;
    compute_strides(ndim, shape, strides);
    total_size = compute_size(ndim, shape);
    cudaMalloc(&data, total_size * sizeof(float));
}

Tensor::Tensor(int d0, int d1, int d2, int d3) {
    ndim = 4;
    shape[0] = d0; shape[1] = d1; shape[2] = d2; shape[3] = d3;
    compute_strides(ndim, shape, strides);
    total_size = compute_size(ndim, shape);
    cudaMalloc(&data, total_size * sizeof(float));
}

Tensor::Tensor(const std::vector<int>& dims) {
    ndim = dims.size();
    if (ndim < 1 || ndim > 4)
        throw std::runtime_error("Tensor: only 1-4 dims supported");
    for (int i = 0; i < ndim; i++) shape[i] = dims[i];
    for (int i = ndim; i < 4; i++) shape[i] = 1;
    compute_strides(ndim, shape, strides);
    total_size = compute_size(ndim, shape);
    cudaMalloc(&data, total_size * sizeof(float));
}

Tensor::~Tensor() {
    if (data) cudaFree(data);
}

// ---- Move semantics ----
Tensor::Tensor(Tensor&& other) noexcept {
    data = other.data;
    ndim = other.ndim;
    total_size = other.total_size;
    for (int i = 0; i < 4; i++) {
        shape[i] = other.shape[i];
        strides[i] = other.strides[i];
    }
    other.data = nullptr;
    other.ndim = 0;
    other.total_size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data) cudaFree(data);
        data = other.data;
        ndim = other.ndim;
        total_size = other.total_size;
        for (int i = 0; i < 4; i++) {
            shape[i] = other.shape[i];
            strides[i] = other.strides[i];
        }
        other.data = nullptr;
        other.ndim = 0;
        other.total_size = 0;
    }
    return *this;
}

// ---- Data transfer ----
void Tensor::toGPU(const float* host_data) {
    cudaMemcpy(data, host_data, total_size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::toCPU(float* host_data) const {
    cudaMemcpy(host_data, data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::fill(float value) {
    int block = 256;
    int grid = get_grid(total_size);
    fill_kernel<<<grid, block>>>(data, value, total_size);
    cudaDeviceSynchronize();
}

void Tensor::print(int limit) const {
    float* host = new float[total_size];
    cudaMemcpy(host, data, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Tensor(";
    for (int i = 0; i < ndim; i++) {
        std::cout << shape[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << ") [";

    for (int i = 0; i < std::min(total_size, limit); i++) {
        std::cout << host[i];
        if (i < std::min(total_size, limit) - 1) std::cout << ", ";
    }
    if (total_size > limit) std::cout << "...";
    std::cout << "]" << std::endl;

    delete[] host;
}

// ---- View (no allocation, share data pointer) ----
Tensor Tensor::view(const std::vector<int>& new_shape) const {
    Tensor t;
    t.data = data;  // SHARED pointer — no copy
    t.ndim = new_shape.size();
    for (int i = 0; i < t.ndim; i++) t.shape[i] = new_shape[i];
    for (int i = t.ndim; i < 4; i++) t.shape[i] = 1;
    compute_strides(t.ndim, t.shape, t.strides);
    t.total_size = compute_size(t.ndim, t.shape);

    if (t.total_size != total_size)
        throw std::runtime_error("View: total elements must match");
    return t;
}

Tensor Tensor::view(int d0, int d1) const {
    return view({d0, d1});
}

Tensor Tensor::view(int d0, int d1, int d2) const {
    return view({d0, d1, d2});
}

Tensor Tensor::view(int d0, int d1, int d2, int d3) const {
    return view({d0, d1, d2, d3});
}

// ---- Slice ----
Tensor Tensor::slice(int axis, int start, int end) const {
    // Only supports contiguous slicing along one axis for simplicity
    if (axis < 0 || axis >= ndim)
        throw std::runtime_error("Slice: invalid axis");

    int slice_len = end - start;
    int block_size = 1;
    for (int i = axis + 1; i < ndim; i++) block_size *= shape[i];

    int prefix = 1;
    for (int i = 0; i < axis; i++) prefix *= shape[i];

    Tensor result;
    result.ndim = ndim;
    for (int i = 0; i < ndim; i++) result.shape[i] = shape[i];
    result.shape[axis] = slice_len;
    compute_strides(result.ndim, result.shape, result.strides);
    result.total_size = compute_size(result.ndim, result.shape);

    cudaMalloc(&result.data, result.total_size * sizeof(float));

    // Copy slices
    int total_blocks = prefix;
    int src_slice_size = block_size * shape[axis];
    int dst_slice_size = block_size * slice_len;

    for (int b = 0; b < total_blocks; b++) {
        int src_offset = b * src_slice_size + start * block_size;
        int dst_offset = b * dst_slice_size;
        cudaMemcpy(result.data + dst_offset, data + src_offset,
                   dst_slice_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return result;
}

// ---- Transpose (only contiguous 2D for matmul purposes) ----
Tensor Tensor::transpose(int axis1, int axis2) const {
    if (ndim < 2)
        throw std::runtime_error("Transpose: need at least 2 dims");

    Tensor result;
    result.ndim = ndim;
    for (int i = 0; i < ndim; i++) result.shape[i] = shape[i];
    std::swap(result.shape[axis1], result.shape[axis2]);
    compute_strides(result.ndim, result.shape, result.strides);
    result.total_size = total_size;

    cudaMalloc(&result.data, result.total_size * sizeof(float));

    // Simple element-by-element transpose (can optimize later)
    float* host_src = new float[total_size];
    float* host_dst = new float[total_size];
    toCPU(host_src);

    int src_strides[4], dst_strides[4];
    compute_strides(ndim, shape, src_strides);
    compute_strides(result.ndim, result.shape, dst_strides);

    // Flattened index mapping
    for (int i = 0; i < total_size; i++) {
        // Convert flat index to multi-index
        int idx[4] = {0};
        int tmp = i;
        for (int d = 0; d < ndim; d++) {
            idx[d] = tmp / src_strides[d];
            tmp %= src_strides[d];
        }
        // Swap axes
        std::swap(idx[axis1], idx[axis2]);
        // Convert back to flat
        int j = 0;
        for (int d = 0; d < ndim; d++) {
            j += idx[d] * dst_strides[d];
        }
        host_dst[j] = host_src[i];
    }

    result.toGPU(host_dst);
    delete[] host_src;
    delete[] host_dst;
    return result;
}

// ---- Copy (allocates new memory) ----
Tensor Tensor::copy() const {
    Tensor result;
    result.ndim = ndim;
    result.total_size = total_size;
    for (int i = 0; i < 4; i++) {
        result.shape[i] = shape[i];
        result.strides[i] = strides[i];
    }
    cudaMalloc(&result.data, total_size * sizeof(float));
    int block = 256, grid = get_grid(total_size);
    copy_kernel<<<grid, block>>>(result.data, data, total_size);
    cudaDeviceSynchronize();
    return result;
}

// ---- Reshape (copies data, new allocation) ----
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    Tensor result(new_shape);
    if (result.total_size != total_size)
        throw std::runtime_error("Reshape: total elements must match");
    cudaMemcpy(result.data, data, total_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
    return result;
}

// ==================== STANDALONE OPS ====================

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.total_size != b.total_size)
        throw std::runtime_error("Add: tensors must have same size");

    Tensor result = a.copy();
    int block = 256, grid = get_grid(a.total_size);
    add_kernel<<<grid, block>>>(result.data, a.data, b.data, a.total_size);
    cudaDeviceSynchronize();
    return result;
}

void add_inplace(Tensor& a, const Tensor& b) {
    if (a.total_size != b.total_size)
        throw std::runtime_error("add_inplace: tensors must have same size");
    int block = 256, grid = get_grid(a.total_size);
    add_kernel<<<grid, block>>>(a.data, a.data, b.data, a.total_size);
    cudaDeviceSynchronize();
}

Tensor scale(const Tensor& a, float scalar) {
    Tensor result = a.copy();
    int block = 256, grid = get_grid(a.total_size);
    scale_kernel<<<grid, block>>>(result.data, a.data, scalar, a.total_size);
    cudaDeviceSynchronize();
    return result;
}

void scale_inplace(Tensor& a, float scalar) {
    int block = 256, grid = get_grid(a.total_size);
    scale_kernel<<<grid, block>>>(a.data, a.data, scalar, a.total_size);
    cudaDeviceSynchronize();
}

Tensor concat(const Tensor& a, const Tensor& b, int axis) {
    if (a.ndim != b.ndim)
        throw std::runtime_error("Concat: tensors must have same ndim");
    if (axis < 0 || axis >= a.ndim)
        throw std::runtime_error("Concat: invalid axis");
    for (int i = 0; i < a.ndim; i++) {
        if (i != axis && a.shape[i] != b.shape[i])
            throw std::runtime_error("Concat: non-concat dims must match");
    }

    std::vector<int> new_shape;
    for (int i = 0; i < a.ndim; i++) {
        new_shape.push_back((i == axis) ? a.shape[i] + b.shape[i] : a.shape[i]);
    }
    Tensor result(new_shape);

    int a_size = a.total_size;
    int b_size = b.total_size;
    int block = 256, grid = get_grid(a_size + b_size);
    concat_kernel<<<grid, block>>>(result.data, a.data, b.data, a_size, b_size);
    cudaDeviceSynchronize();
    return result;
}

// ---- Matmul (simple 2D for now; batched uses this internally) ----
Tensor matmul(const Tensor& a, const Tensor& b) {
    // Expect 2D: a [M, K], b [K, N] → result [M, N]
    if (a.ndim < 2 || b.ndim < 2)
        throw std::runtime_error("Matmul: need at least 2D tensors");
    int M = a.shape[0];
    int K = a.shape[1];
    int Kb = b.shape[0];
    int N = b.shape[1];
    if (K != Kb)
        throw std::runtime_error("Matmul: inner dims must match");

    Tensor result(M, N);
    // Use cuBLAS or simple CPU fallback for prototype
    // Simple triple-loop on host (REPLACE with cuBLAS in production!)
    float* host_a = new float[a.total_size];
    float* host_b = new float[b.total_size];
    float* host_c = new float[result.total_size]();

    a.toCPU(host_a);
    b.toCPU(host_b);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += host_a[i * K + k] * host_b[k * N + j];
            }
            host_c[i * N + j] = sum;
        }
    }

    result.toGPU(host_c);
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;
    return result;
}

// ---- Batched matmul ----
Tensor batched_matmul(const Tensor& a, const Tensor& b) {
    // a: [B, M, K], b: [B, K, N] → [B, M, N]
    if (a.ndim != 3 || b.ndim != 3)
        throw std::runtime_error("BatchedMatmul: expect 3D tensors");

    int B = a.shape[0];
    int M = a.shape[1];
    int K = a.shape[2];
    int Kb = b.shape[1];
    int N = b.shape[2];

    if (B != b.shape[0] || K != Kb)
        throw std::runtime_error("BatchedMatmul: shape mismatch");

    Tensor result(B, M, N);

    float* host_a = new float[a.total_size];
    float* host_b = new float[b.total_size];
    float* host_c = new float[result.total_size]();

    a.toCPU(host_a);
    b.toCPU(host_b);

    for (int b_idx = 0; b_idx < B; b_idx++) {
        int a_off = b_idx * M * K;
        int b_off = b_idx * K * N;
        int c_off = b_idx * M * N;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += host_a[a_off + i * K + k] * host_b[b_off + k * N + j];
                }
                host_c[c_off + i * N + j] = sum;
            }
        }
    }

    result.toGPU(host_c);
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;
    return result;
}