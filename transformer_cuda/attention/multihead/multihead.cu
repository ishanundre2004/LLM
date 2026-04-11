#include <cuda_runtime.h>
#include <math.h>
#include "multihead.h"

__global__ void attention_kernel(float* Q, float* K, float* V, float* O,
                                 int seq_len, int d_model, int num_heads) {
    int d_k = d_model / num_heads;
    int head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (head >= num_heads || row >= seq_len) return;

    // Offset for this head
    int head_offset = head * d_k;
    // Q, K, V are (seq_len, d_model) row-major
    // For each head, we operate on a slice of size (seq_len, d_k)

    float sum_exp = 0.0f;
    float out_tmp[128]; // max d_k = 128 for safety
    for (int i = 0; i < d_k; ++i) out_tmp[i] = 0.0f;

    for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < d_k; ++k) {
            float q_val = Q[row * d_model + head_offset + k];
            float k_val = K[j * d_model + head_offset + k];
            dot += q_val * k_val;
        }
        float att = expf(dot / sqrtf((float)d_k));
        for (int k = 0; k < d_k; ++k) {
            out_tmp[k] += att * V[j * d_model + head_offset + k];
        }
        sum_exp += att;
    }

    for (int k = 0; k < d_k; ++k) {
        O[row * d_model + head_offset + k] = out_tmp[k] / sum_exp;
    }
}

void multihead_attention(Tensor& Q, Tensor& K, Tensor& V, Tensor& output, int num_heads) {
    int seq_len = Q.rows;
    int d_model = Q.cols;
    dim3 threads(1, 1, 1);
    dim3 blocks(1, seq_len, num_heads);
    attention_kernel<<<blocks, threads>>>(Q.data, K.data, V.data, output.data,
                                          seq_len, d_model, num_heads);
    cudaDeviceSynchronize();
}