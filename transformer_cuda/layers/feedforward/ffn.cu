#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#include "ffn.h"
#include "../../tensor/tensor.h"
#include "../../kernels/gelu/gelu.h"
#include "../../kernels/relu/relu.h"

//////////////////////////////////////////////////////////////
// BIAS ADDITION KERNEL
//////////////////////////////////////////////////////////////

__global__ void add_bias_kernel(
    float* data,
    const float* bias,
    int rows,
    int cols
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= rows * cols) return;

    int col = idx % cols;
    data[idx] += bias[col];
}

void addBias(Tensor& t, Tensor& bias)
{
    int size = t.rows * t.cols;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    add_bias_kernel<<<blocks, threads>>>(
        t.data,
        bias.data,
        t.rows,
        t.cols
    );

    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////////////////
// FEEDFORWARD LAYER
//////////////////////////////////////////////////////////////

void feedforward(
    Tensor& X,
    Tensor& W1,
    Tensor& b1,
    Tensor& W2,
    Tensor& b2,
    Tensor& output
)
{
    int seq_len = X.rows;
    int d_model = X.cols;
    int d_ff    = W1.cols;

    Tensor H(seq_len, d_ff);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta  = 0.0f;

    //////////////////////////////////////////////////////////
    // H = X × W1
    //////////////////////////////////////////////////////////
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_ff,       // cols of W1
        seq_len,
        d_model,
        &alpha,
        W1.data,
        d_ff,
        X.data,
        d_model,
        &beta,
        H.data,
        d_ff
    );

    //////////////////////////////////////////////////////////
    // H += b1
    //////////////////////////////////////////////////////////
    addBias(H, b1);

    //////////////////////////////////////////////////////////
    // Activation
    //////////////////////////////////////////////////////////
    gelu(H);
    // relu(H); // optional

    //////////////////////////////////////////////////////////
    // output = H × W2
    //////////////////////////////////////////////////////////
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,
        seq_len,
        d_ff,
        &alpha,
        W2.data,
        d_model,
        H.data,
        d_ff,
        &beta,
        output.data,
        d_model
    );

    //////////////////////////////////////////////////////////
    // output += b2
    //////////////////////////////////////////////////////////
    addBias(output, b2);

    cublasDestroy(handle);
}