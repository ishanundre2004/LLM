#include <cuda_runtime.h>
#include <math.h>
#include "multihead.h"

// ============================================================
// EXISTING KERNELS (UNCHANGED)
// ============================================================

__global__ void split_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int d_model,
    int num_heads,
    int d_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * d_model;
    
    if (idx >= total_elements) return;
    
    int row = idx / d_model;
    int col = idx % d_model;
    int head = col / d_k;
    int k = col % d_k;
    
    output[row * num_heads * d_k + head * d_k + k] = input[idx];
}

__global__ void concat_heads_kernel(
    float* input,
    float* output,
    int seq_len,
    int num_heads,
    int d_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * num_heads * d_k;
    
    if (idx >= total_elements) return;
    
    int row = idx / (num_heads * d_k);
    int remainder = idx % (num_heads * d_k);
    int head = remainder / d_k;
    int k = remainder % d_k;
    
    output[row * num_heads * d_k + head * d_k + k] = input[idx];
}

__global__ void mha_attention_kernel(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int d_k,
    int num_heads
) {
    int head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (head >= num_heads || row >= seq_len) return;

    int head_offset = head * d_k;
    float scale = 1.0f / sqrtf((float)d_k);
    
    extern __shared__ float shared_scores[];
    
    float out_tmp[128] = {0.0f};
    float max_score = -INFINITY;

    // Compute scores
    for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < d_k; ++k) {
            float q_val = Q[row * num_heads * d_k + head_offset + k];
            float k_val = K[j * num_heads * d_k + head_offset + k];
            dot += q_val * k_val;
        }
        dot *= scale;
        shared_scores[j] = dot;
        max_score = fmaxf(max_score, dot);
    }

    // Stable softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float exp_val = expf(shared_scores[j] - max_score);
        shared_scores[j] = exp_val;
        sum_exp += exp_val;
    }

    // Weighted sum
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int j = 0; j < seq_len; ++j) {
        float att = shared_scores[j] * inv_sum;
        for (int k = 0; k < d_k; ++k) {
            out_tmp[k] += att * V[j * num_heads * d_k + head_offset + k];
        }
    }

    // Write output
    for (int k = 0; k < d_k; ++k) {
        O[row * num_heads * d_k + head_offset + k] = out_tmp[k];
    }
}

// ============================================================
// NEW KERNELS WITH CAUSAL MASKING
// ============================================================

__global__ void mha_attention_causal_kernel(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int d_k,
    int num_heads
) {
    int head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (head >= num_heads || row >= seq_len) return;

    int head_offset = head * d_k;
    float scale = 1.0f / sqrtf((float)d_k);
    
    extern __shared__ float shared_scores[];
    
    float out_tmp[128] = {0.0f};
    float max_score = -INFINITY;

    // CAUSAL: Only compute scores for j <= row
    for (int j = 0; j <= row; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < d_k; ++k) {
            float q_val = Q[row * num_heads * d_k + head_offset + k];
            float k_val = K[j * num_heads * d_k + head_offset + k];
            dot += q_val * k_val;
        }
        dot *= scale;
        shared_scores[j] = dot;
        max_score = fmaxf(max_score, dot);
    }
    
    // Set future positions to -infinity
    for (int j = row + 1; j < seq_len; ++j) {
        shared_scores[j] = -INFINITY;
    }

    // Stable softmax (future positions with -inf become 0)
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float exp_val = (shared_scores[j] == -INFINITY) ? 
                        0.0f : expf(shared_scores[j] - max_score);
        shared_scores[j] = exp_val;
        sum_exp += exp_val;
    }

    // Weighted sum (only attended positions contribute)
    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
    for (int j = 0; j <= row; ++j) {
        float att = shared_scores[j] * inv_sum;
        for (int k = 0; k < d_k; ++k) {
            out_tmp[k] += att * V[j * num_heads * d_k + head_offset + k];
        }
    }

    // Write output
    for (int k = 0; k < d_k; ++k) {
        O[row * num_heads * d_k + head_offset + k] = out_tmp[k];
    }
}

__global__ void mha_attention_masked_kernel(
    float* Q,
    float* K,
    float* V,
    float* mask,    // seq_len x seq_len mask tensor
    float* O,
    int seq_len,
    int d_k,
    int num_heads
) {
    int head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (head >= num_heads || row >= seq_len) return;

    int head_offset = head * d_k;
    float scale = 1.0f / sqrtf((float)d_k);
    
    extern __shared__ float shared_scores[];
    
    float out_tmp[128] = {0.0f};
    float max_score = -INFINITY;

    // Compute scores with mask
    for (int j = 0; j < seq_len; ++j) {
        float mask_val = mask[row * seq_len + j];
        
        if (mask_val > 0.0f) {
            float dot = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                float q_val = Q[row * num_heads * d_k + head_offset + k];
                float k_val = K[j * num_heads * d_k + head_offset + k];
                dot += q_val * k_val;
            }
            dot *= scale;
            shared_scores[j] = dot;
            max_score = fmaxf(max_score, dot);
        } else {
            shared_scores[j] = -INFINITY;
        }
    }

    // Stable softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float exp_val = (shared_scores[j] == -INFINITY) ? 
                        0.0f : expf(shared_scores[j] - max_score);
        shared_scores[j] = exp_val;
        sum_exp += exp_val;
    }

    // Weighted sum
    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        if (shared_scores[j] > 0.0f) {
            float att = shared_scores[j] * inv_sum;
            for (int k = 0; k < d_k; ++k) {
                out_tmp[k] += att * V[j * num_heads * d_k + head_offset + k];
            }
        }
    }

    // Write output
    for (int k = 0; k < d_k; ++k) {
        O[row * num_heads * d_k + head_offset + k] = out_tmp[k];
    }
}

// ============================================================
// EXISTING HOST FUNCTIONS (UNCHANGED)
// ============================================================

void split_heads(Tensor& input, Tensor& output, int seq_len, int d_model, int num_heads, int d_k) {
    int total_elements = seq_len * d_model;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    split_heads_kernel<<<blocks, threads>>>(
        input.data, output.data, seq_len, d_model, num_heads, d_k
    );
    cudaDeviceSynchronize();
}

void concat_heads(Tensor& input, Tensor& output, int seq_len, int num_heads, int d_k) {
    int total_elements = seq_len * num_heads * d_k;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    concat_heads_kernel<<<blocks, threads>>>(
        input.data, output.data, seq_len, num_heads, d_k
    );
    cudaDeviceSynchronize();
}

void multihead_attention(Tensor& Q, Tensor& K, Tensor& V, Tensor& output, int num_heads) {
    int seq_len = Q.rows;
    int d_model = Q.cols;
    int d_k = d_model / num_heads;
    
    dim3 threads(1, 1, 1);
    dim3 blocks(1, seq_len, num_heads);
    size_t shared_mem = seq_len * sizeof(float);
    
    mha_attention_kernel<<<blocks, threads, shared_mem>>>(
        Q.data, K.data, V.data, output.data,
        seq_len, d_k, num_heads
    );
    cudaDeviceSynchronize();
}

// ============================================================
// NEW HOST FUNCTIONS FOR MASKING
// ============================================================

void multihead_attention_causal(Tensor& Q, Tensor& K, Tensor& V, Tensor& output, int num_heads) {
    int seq_len = Q.rows;
    int d_model = Q.cols;
    int d_k = d_model / num_heads;
    
    dim3 threads(1, 1, 1);
    dim3 blocks(1, seq_len, num_heads);
    size_t shared_mem = seq_len * sizeof(float);
    
    mha_attention_causal_kernel<<<blocks, threads, shared_mem>>>(
        Q.data, K.data, V.data, output.data,
        seq_len, d_k, num_heads
    );
    cudaDeviceSynchronize();
}

void multihead_attention_masked(
    Tensor& Q, Tensor& K, Tensor& V, 
    Tensor& mask, Tensor& output, int num_heads
) {
    int seq_len = Q.rows;
    int d_model = Q.cols;
    int d_k = d_model / num_heads;
    
    dim3 threads(1, 1, 1);
    dim3 blocks(1, seq_len, num_heads);
    size_t shared_mem = seq_len * sizeof(float);
    
    mha_attention_masked_kernel<<<blocks, threads, shared_mem>>>(
        Q.data, K.data, V.data, mask.data, output.data,
        seq_len, d_k, num_heads
    );
    cudaDeviceSynchronize();
}