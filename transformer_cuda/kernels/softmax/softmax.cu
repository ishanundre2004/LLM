#include "softmax.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>

__global__ void softmax_kernel(float *input, float *output, int rows, int cols)
{
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows)
        return;

    float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    // ---- MAX REDUCTION ----
    float max_val = -FLT_MAX;

    for (int i = tid; i < cols; i += blockDim.x)
    {
        max_val = fmaxf(max_val, row_ptr[i]);
    }

    shared[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float max_val_global = shared[0];

    // ---- EXP + SUM ----
    float sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x)
    {
        float val = expf(row_ptr[i] - max_val_global);
        out_ptr[i] = val;
        sum += val;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float sum_global = shared[0];

    // ---- NORMALIZE ----
    for (int i = tid; i < cols; i += blockDim.x)
    {
        out_ptr[i] /= sum_global;
    }
}

void softmax(Tensor& input, Tensor& output, int rows, int cols)
{
    int threads = min(1024, cols);
    int blocks = rows;
    size_t shared_mem = threads * sizeof(float);

    softmax_kernel<<<blocks, threads, shared_mem>>>(
        input.data,
        output.data,
        rows,
        cols
    );
    cudaDeviceSynchronize();
}

void softmax(Tensor& input, Tensor& output)
{
    softmax(input, output, input.rows, input.cols);
}
