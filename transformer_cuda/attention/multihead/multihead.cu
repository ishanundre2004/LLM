#include <cuda_runtime.h>
#include "multihead.h"
#include <cmath>

__global__ void split_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int d_model,
    int num_heads,
    int d_k
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx < total){
        int row = idx / d_model;
        int col = idx % d_model;

        int head = col / d_k;
        int head_col = col % d_k;

        int out_idx =
            head * (seq_len * d_k) +
            row * d_k +
            head_col;

        output[out_idx] = input[idx];
    }
}

__global__ void concat_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int num_heads,
    int d_k
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = seq_len * num_heads * d_k;

    if (idx < total){
        int head = idx / (seq_len * d_k);
        int rem = idx % (seq_len * d_k);

        int row = rem / d_k;
        int head_col = rem % d_k;

        int out_col = head * d_k + head_col;
        int out_idx = row * (num_heads * d_k) + out_col;

        output[out_idx] = input[idx];
    }
}

__global__ void mha_attention_kernel(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int d_k,
    int num_heads
)
{
    int head = blockIdx.x / seq_len;
    int row  = blockIdx.x % seq_len;

    if (head >= num_heads || row >= seq_len) return;

    int offset = head * seq_len * d_k;

    float* Q_h = Q + offset;
    float* K_h = K + offset;
    float* V_h = V + offset;
    float* O_h = O + offset;

    float sum = 0.0f;

    // initialize output row
    for (int k = 0; k < d_k; k++)
        O_h[row * d_k + k] = 0.0f;

    for (int j = 0; j < seq_len; j++)
    {
        float dot = 0.0f;

        for (int k = 0; k < d_k; k++)
            dot += Q_h[row * d_k + k] * K_h[j * d_k + k];

        float val = expf(dot);

        for (int k = 0; k < d_k; k++)
            O_h[row * d_k + k] += val * V_h[j * d_k + k];

        sum += val;
    }

    // normalize
    for (int k = 0; k < d_k; k++)
        O_h[row * d_k + k] /= sum;
}





void split_heads(
    Tensor& input,
    Tensor& output,
    int seq_len,
    int d_model,
    int num_heads,
    int d_k
)
{
    int total = seq_len * d_model;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    split_heads_kernel<<<blocks, threads>>>(
        input.data,
        output.data,
        seq_len,
        d_model,
        num_heads,
        d_k
    );
}

void concat_heads(
    Tensor& input,
    Tensor& output,
    int seq_len,
    int num_heads,
    int d_k
)
{
    int total = seq_len * num_heads * d_k;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    concat_heads_kernel<<<blocks, threads>>>(
        input.data,
        output.data,
        seq_len,
        num_heads,
        d_k
    );
}

void multihead_attention(
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& output,
    int num_heads
)
{
    int seq_len = Q.rows;
    int d_model = Q.cols;
    int d_k = d_model / num_heads;

    Tensor Qh(num_heads * seq_len, d_k);
    Tensor Kh(num_heads * seq_len, d_k);
    Tensor Vh(num_heads * seq_len, d_k);
    Tensor Oh(num_heads * seq_len, d_k);

    split_heads(Q, Qh, seq_len, d_model, num_heads, d_k);
    split_heads(K, Kh, seq_len, d_model, num_heads, d_k);
    split_heads(V, Vh, seq_len, d_model, num_heads, d_k);

    int blocks = num_heads * seq_len;
    int threads = 1;

    mha_attention_kernel<<<blocks, threads>>>(
        Qh.data,
        Kh.data,
        Vh.data,
        Oh.data,
        seq_len,
        d_k,
        num_heads
    );

    concat_heads(Oh, output, seq_len, num_heads, d_k);
}
