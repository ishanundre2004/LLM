#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "qkv.h"

void compute_qkv(
    Tensor& X,
    Tensor& Wqkv,
    Tensor& Q,
    Tensor& K,
    Tensor& V
)
{
    int seq_len = X.rows;
    int d_model = X.cols;
    int d_k = Q.cols;

    Tensor QKV(seq_len, 3 * d_k);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // C = A × B
    // (seq_len × d_model) × (d_model × 3*d_k)

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        3 * d_k,
        seq_len,
        d_model,
        &alpha,
        Wqkv.data,
        3 * d_k,
        X.data,
        d_model,
        &beta,
        QKV.data,
        3 * d_k
    );

    // Split QKV → Q, K, V

    int total = seq_len * d_k;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    split_qkv_kernel<<<blocks, threads>>>(
        QKV.data,
        Q.data,
        K.data,
        V.data,
        seq_len,
        d_k
    );

    cudaDeviceSynchronize();

    cublasDestroy(handle);
}

__global__ void split_qkv_kernel(
    float* qkv,
    float* Q,
    float* K,
    float* V,
    int seq_len,
    int d_k
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_k;

    if (idx >= total) return;

    int row = idx / d_k;
    int col = idx % d_k;

    int base = row * (3 * d_k);

    Q[idx] = qkv[base + col];
    K[idx] = qkv[base + d_k + col];
    V[idx] = qkv[base + 2*d_k + col];
}