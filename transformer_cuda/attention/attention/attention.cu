#include "attention.h"
#include "../../kernels/softmax/softmax.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <iostream>

namespace
{

    __global__ void scale_kernel(float *data, int size, float scale)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size)
            data[idx] *= scale;
    }

}
void compute_scores(
    Tensor &Q,
    Tensor &K,
    Tensor &S,
    cublasHandle_t handle)
{
    int seq_len = Q.rows;
    int d_k = Q.cols;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_T, // K^T
        CUBLAS_OP_N,
        seq_len,
        seq_len,
        d_k,
        &alpha,
        K.data,
        d_k,
        Q.data,
        d_k,
        &beta,
        S.data,
        seq_len);
}

void scale_tensor(Tensor &S, float d_k)
{
    int size = S.rows * S.cols;

    float scale = 1.0f / sqrtf(d_k);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    scale_kernel<<<blocks, threads>>>(S.data, size, scale);
}

void compute_output(
    Tensor &P,
    Tensor &V,
    Tensor &O,
    cublasHandle_t handle)
{
    int seq_len = P.rows;
    int d_k = V.cols;

    float alpha = 1.0f;
    float beta = 0.0f;

    // NOTE:
    // Computes O = P * V (row-major equivalent via cuBLAS)

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_k,
        seq_len,
        seq_len,
        &alpha,
        V.data,
        d_k,
        P.data,
        seq_len,
        &beta,
        O.data,
        d_k);
}

void attention(
    Tensor &Q,
    Tensor &K,
    Tensor &V,
    Tensor &O)
{
    if (Q.rows != K.rows || Q.cols != K.cols)
    {
        std::cerr << "Q and K shape mismatch\n";
        return;
    }

    if (Q.rows != V.rows)
    {
        std::cerr << "Q and V sequence mismatch\n";
        return;
    }

    int seq_len = Q.rows;
    int d_k = Q.cols;

    Tensor S(seq_len, seq_len);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // -------- Step 1: Scores --------
    compute_scores(Q, K, S, handle);

    // -------- Step 2: Scale --------
    scale_tensor(S, d_k);

    // -------- Step 3: Softmax --------
    softmax(S, S);

    // -------- Step 4: Output --------
    compute_output(S, V, O, handle);

    // -------- Cleanup --------
    cublasDestroy(handle);
}