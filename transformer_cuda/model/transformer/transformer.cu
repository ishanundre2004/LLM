#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include "transformer.h"

//////////////////////////////////////////////////////////////
// ARGMAX KERNEL
//////////////////////////////////////////////////////////////

__global__ void argmax_kernel(
    float* logits,
    int* output,
    int rows,
    int cols
)
{
    int row = blockIdx.x;

    if (row >= rows) return;

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int i = 0; i < cols; i++)
    {
        float val = logits[row * cols + i];
        if (val > max_val)
        {
            max_val = val;
            max_idx = i;
        }
    }

    output[row] = max_idx;
}

void argmax(Tensor& logits, int* output_tokens)
{
    int rows = logits.rows;
    int cols = logits.cols;

    argmax_kernel<<<rows, 1>>>(
        logits.data,
        output_tokens,
        rows,
        cols
    );
}

//////////////////////////////////////////////////////////////
// TRANSFORMER FORWARD
//////////////////////////////////////////////////////////////

void transformer_forward(
    int* tokens,
    Tensor& embedding,
    Tensor* Wqkv,
    Tensor* W1,
    Tensor* W2,
    Tensor& W_vocab,
    Tensor& output,
    int num_layers,
    int num_heads,
    int seq_len
)
{
    int d_model = embedding.cols;

    //////////////////////////////////////////////////////////
    // cuBLAS SETUP (reuse in real systems!)
    //////////////////////////////////////////////////////////
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta  = 0.0f;

    //////////////////////////////////////////////////////////
    // STEP 1 — Embedding + Positional Encoding
    //////////////////////////////////////////////////////////
    Tensor X(seq_len, d_model);

    fused_embedding_positional(
        embedding,
        tokens,
        X,
        seq_len
    );

    //////////////////////////////////////////////////////////
    // STEP 2 — Transformer Layers
    //////////////////////////////////////////////////////////
    Tensor temp(seq_len, d_model);

    for (int i = 0; i < num_layers; i++)
    {
        transformer_block(
            X,
            Wqkv[i],
            W1[i],
            W2[i],
            temp,
            num_heads
        );

        // Swap instead of reallocating (IMPORTANT optimization)
        std::swap(X.data, temp.data);
    }

    //////////////////////////////////////////////////////////
    // STEP 3 — Final Projection (Logits)
    //////////////////////////////////////////////////////////
    // (seq_len x d_model) * (d_model x vocab)
    // = (seq_len x vocab)

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        W_vocab.cols,   // vocab_size
        seq_len,
        d_model,
        &alpha,
        W_vocab.data,
        W_vocab.cols,
        X.data,
        d_model,
        &beta,
        output.data,
        W_vocab.cols
    );

    //////////////////////////////////////////////////////////
    // OPTIONAL — Softmax (for probabilities)
    //////////////////////////////////////////////////////////
    // softmax(output, output);

    cublasDestroy(handle);
}

//////////////////////////////////////////////////////////////
// OPTIONAL: SIMPLE INFERENCE LOOP
//////////////////////////////////////////////////////////////

void generate(
    int* tokens,
    int max_len,
    Tensor& embedding,
    Tensor* Wqkv,
    Tensor* W1,
    Tensor* W2,
    Tensor& W_vocab,
    int num_layers,
    int num_heads,
    int vocab_size
)
{
    Tensor logits(max_len, vocab_size);

    int* d_output_tokens;
    cudaMalloc(&d_output_tokens, max_len * sizeof(int));

    for (int t = 0; t < max_len; t++)
    {
        transformer_forward(
            tokens,
            embedding,
            Wqkv,
            W1,
            W2,
            W_vocab,
            logits,
            num_layers,
            num_heads,
            t + 1   // growing sequence
        );

        // Argmax over last row only (optimization possible)
        argmax(logits, d_output_tokens);

        int next_token;
        cudaMemcpy(
            &next_token,
            d_output_tokens + t,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );

        tokens[t + 1] = next_token;
    }

    cudaFree(d_output_tokens);
}