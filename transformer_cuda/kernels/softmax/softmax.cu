#include "softmax.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_kernel_stable(
    float *input, 
    float *output, 
    int rows, 
    int cols,
    float scale
) {
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    float *sdata = shared;  // Single shared memory array for all reductions

    // STEP 1: Find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = row_ptr[i] * scale;
        max_val = fmaxf(max_val, val);
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Parallel max reduction - ALL threads participate
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float global_max = sdata[0];
    __syncthreads();

    // STEP 2: Compute exp and sum
    float sum = 0.0f;
    float comp = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(row_ptr[i] * scale - global_max);
        out_ptr[i] = val;
        
        // Kahan summation
        float y = val - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Parallel sum reduction - ALL threads participate
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float global_sum = sdata[0];
    __syncthreads();

    // STEP 3: Normalize
    if (global_sum > 0.0f) {
        float inv_sum = 1.0f / global_sum;
        for (int i = tid; i < cols; i += blockDim.x) {
            out_ptr[i] *= inv_sum;
        }
    }
}

// Host functions
void softmax(Tensor& input, Tensor& output, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    
    int threads = 256;
    while (threads > cols) threads /= 2;
    if (threads < 32) threads = 32;
    
    int blocks = rows;
    size_t shared_mem = threads * sizeof(float);

    softmax_kernel_stable<<<blocks, threads, shared_mem>>>(
        input.data, output.data, rows, cols, 1.0f
    );
    cudaDeviceSynchronize();
}

void softmax(Tensor& input, Tensor& output) {
    softmax(input, output, input.rows, input.cols);
}

void softmax_scaled(Tensor& input, Tensor& output, float scale) {
    if (input.rows <= 0 || input.cols <= 0) return;
    
    int threads = 256;
    while (threads > input.cols) threads /= 2;
    if (threads < 32) threads = 32;
    
    int blocks = input.rows;
    size_t shared_mem = threads * sizeof(float);

    softmax_kernel_stable<<<blocks, threads, shared_mem>>>(
        input.data, output.data, input.rows, input.cols, scale
    );
    cudaDeviceSynchronize();
}

void softmax_inplace(Tensor& input, float scale) {
    // Use same kernel but with output = input
    if (input.rows <= 0 || input.cols <= 0) return;
    
    Tensor temp(input.rows, input.cols);
    softmax_scaled(input, temp, scale);
    
    // Copy back (simple approach, you can optimize later)
    cudaMemcpy(input.data, temp.data, input.rows * input.cols * sizeof(float), cudaMemcpyDeviceToDevice);
}