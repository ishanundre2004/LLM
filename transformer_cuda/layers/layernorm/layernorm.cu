#include <cuda_runtime.h>
#include <math.h>
#include "layernorm.h"

#define EPS 1e-5f

__global__ void layernorm_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    extern __shared__ float shared[];
    float* s_mean = shared;
    float* s_var  = shared + blockDim.x;

    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    // --------------------
    // 1. Compute mean
    // --------------------
    float sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x)
        sum += row_ptr[i];

    s_mean[tid] = sum;
    __syncthreads();

    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_mean[tid] += s_mean[tid + stride];
        __syncthreads();
    }

    float mean = s_mean[0] / cols;

    // --------------------
    // 2. Compute variance
    // --------------------
    float var = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x)
    {
        float diff = row_ptr[i] - mean;
        var += diff * diff;
    }

    s_var[tid] = var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_var[tid] += s_var[tid + stride];
        __syncthreads();
    }

    float variance = s_var[0] / cols;

    // --------------------
    // 3. Normalize
    // --------------------
    float inv_std = rsqrtf(variance + EPS);

    for (int i = tid; i < cols; i += blockDim.x)
    {
        out_ptr[i] = (row_ptr[i] - mean) * inv_std;
    }
}

void layernorm(Tensor& input, Tensor& output)
{
    int rows = input.rows;
    int cols = input.cols;

    int threads = 256;
    int blocks = rows;

    size_t shared_mem = 2 * threads * sizeof(float);

    layernorm_kernel<<<blocks, threads, shared_mem>>>(
        input.data,
        output.data,
        rows,
        cols
    );

    cudaDeviceSynchronize();
}