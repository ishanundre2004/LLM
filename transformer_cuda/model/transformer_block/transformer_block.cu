#include "transformer_block.h"

// Core tensor
#include "../../tensor/tensor.h"

// Attention
#include "../../attention/qkv/qkv.h"
#include "../../attention/multihead/multihead.h"

// Layers
#include "../../layers/layernorm/layernorm.h"
#include "../../layers/feedforward/ffn.h"
#include "../../layers/residual_add/residual_add.h"

// CUDA
#include <cuda_runtime.h>

//////////////////////////////////////////////////////////////
// TRANSFORMER BLOCK
//////////////////////////////////////////////////////////////

void transformer_block(
    Tensor& X,
    Tensor& Wqkv,
    Tensor& W1,
    Tensor& W2,
    Tensor& output,
    int num_heads
)
{
    int seq_len = X.rows;
    int d_model = X.cols;

    //////////////////////////////////////////////////////////
    // QKV Projection
    //////////////////////////////////////////////////////////
    Tensor Q(seq_len, d_model);
    Tensor K(seq_len, d_model);
    Tensor V(seq_len, d_model);

    compute_qkv(X, Wqkv, Q, K, V);

    //////////////////////////////////////////////////////////
    // Multi-Head Attention
    //////////////////////////////////////////////////////////
    Tensor mha_out(seq_len, d_model);

    multihead_attention(Q, K, V, mha_out, num_heads);

    //////////////////////////////////////////////////////////
    // Residual + LayerNorm (Post-Norm)
    //////////////////////////////////////////////////////////
    residual_add(mha_out, X);

    Tensor norm1(seq_len, d_model);
    layernorm(mha_out, norm1);

    //////////////////////////////////////////////////////////
    // Feed Forward Network
    //////////////////////////////////////////////////////////
    Tensor ffn_out(seq_len, d_model);

    // ⚠️ IMPORTANT: Your FFN expects biases
    // You must pass b1 and b2 — currently missing in your design

    // TEMP FIX (if you haven't added biases yet):
    Tensor b1(1, W1.cols);   // dummy bias
    Tensor b2(1, W2.cols);   // dummy bias

    feedforward(norm1, W1, b1, W2, b2, ffn_out);

    //////////////////////////////////////////////////////////
    // Residual + LayerNorm
    //////////////////////////////////////////////////////////
    residual_add(ffn_out, norm1);

    layernorm(ffn_out, output);
}