#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include <assert.h>
#include "transformer.h"

// REMOVED: matmul_kernel and matmul functions (they're now in qkv.cu)

void transformer_forward(
    int* tokens,
    Tensor& embedding,
    Tensor** Wqkv,
    Tensor** W1, Tensor** b1,
    Tensor** W2, Tensor** b2,
    Tensor& W_vocab,
    Tensor& output,
    int num_layers,
    int num_heads,
    int seq_len)
{
    int d_model = embedding.cols;

    // Step 1: Embedding + positional encoding
    Tensor X(seq_len, d_model);
    fused_embedding_positional(embedding, tokens, X, seq_len);

    Tensor temp(seq_len, d_model);

    // Step 2: Transformer layers
    for (int l = 0; l < num_layers; ++l) {
        transformer_block(X, *Wqkv[l], *W1[l], *b1[l], *W2[l], *b2[l], temp, num_heads);
        std::swap(X.data, temp.data);
    }

    // Step 3: Final projection using matmul from qkv.cu
    matmul(X, W_vocab, output);  // matmul is declared in qkv.h
}