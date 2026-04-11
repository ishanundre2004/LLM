#include "transformer_block.h"
#include "../../attention/qkv/qkv.h"
#include "../../attention/multihead/multihead.h"
#include "../../layers/layernorm/layernorm.h"
#include "../../layers/feedforward/ffn.h"
#include "../../layers/residual_add/residual_add.h"

void transformer_block(
    Tensor& X,
    Tensor& Wqkv,
    Tensor& W1, Tensor& b1,
    Tensor& W2, Tensor& b2,
    Tensor& output,
    int num_heads)
{
    int seq_len = X.rows;
    int d_model = X.cols;

    // QKV projection: X (seq_len,d_model) * Wqkv (d_model,3*d_model) -> QKV (seq_len,3*d_model)
    Tensor QKV(seq_len, 3 * d_model);
    matmul(X, Wqkv, QKV);   // we'll define matmul in qkv.cu

    // Split QKV into Q, K, V each (seq_len, d_model)
    Tensor Q(seq_len, d_model), K(seq_len, d_model), V(seq_len, d_model);
    split_qkv(QKV, Q, K, V);   // new kernel – see below

    // Multi-head attention
    Tensor attn_out(seq_len, d_model);
    multihead_attention(Q, K, V, attn_out, num_heads);

    // Residual + LayerNorm (post-norm)
    residual_add(attn_out, X);     // attn_out = attn_out + X
    Tensor norm1(seq_len, d_model);
    layernorm(attn_out, norm1);    // norm1 = LayerNorm(attn_out)

    // Feed-forward network
    Tensor ffn_out(seq_len, d_model);
    feedforward(norm1, W1, b1, W2, b2, ffn_out);

    // Residual + LayerNorm
    residual_add(ffn_out, norm1);
    layernorm(ffn_out, output);
}