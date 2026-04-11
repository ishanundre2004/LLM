#include <cuda_runtime.h>
#include <assert.h>
#include "qkv.h"

// Row-major matrix multiply (reuse from transformer.cu or keep here)
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void matmul(Tensor& A, Tensor& B, Tensor& C) {
    int M = A.rows, K = A.cols, N = B.cols;
    assert(B.rows == K);
    C.rows = M; C.cols = N;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    matmul_kernel<<<grid, block>>>(A.data, B.data, C.data, M, N, K);
    cudaDeviceSynchronize();
}

__global__ void split_qkv_kernel(float* qkv, float* Q, float* K, float* V,
                                 int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;
    int row = idx / d_model;
    int col = idx % d_model;
    Q[idx] = qkv[row * (3*d_model) + col];
    K[idx] = qkv[row * (3*d_model) + d_model + col];
    V[idx] = qkv[row * (3*d_model) + 2*d_model + col];
}

void split_qkv(Tensor& QKV, Tensor& Q, Tensor& K, Tensor& V) {
    int seq_len = QKV.rows;
    int d_model = QKV.cols / 3;   // because QKV is (seq_len, 3*d_model)
    int total = seq_len * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    split_qkv_kernel<<<blocks, threads>>>(QKV.data, Q.data, K.data, V.data,
                                          seq_len, d_model);
    cudaDeviceSynchronize();
}

void compute_qkv(Tensor& X, Tensor& Wqkv, Tensor& Q, Tensor& K, Tensor& V) {
    // X: (seq_len, d_model), Wqkv: (d_model, 3*d_model)
    Tensor QKV(X.rows, 3 * X.cols);
    matmul(X, Wqkv, QKV);
    split_qkv(QKV, Q, K, V);
}