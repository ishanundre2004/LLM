#include <cmath>
#include <iostream>
#include <vector>
#include "include/test_utils.h"
#include "../tensor/tensor.h"
#include "../model/transformer/transformer.h"

bool test_transformer(){
    int seq_len = 3;
    int vocab_size = 7;
    int d_model = 8;
    int d_ff = 32;
    int num_heads = 2;
    int num_layers = 1;

    int tokens_host[3] = {0,1,2};
    int* d_tokens = nullptr;
    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMemcpy(d_tokens, tokens_host, seq_len * sizeof(int), cudaMemcpyHostToDevice);

    Tensor embedding(vocab_size, d_model);
    Tensor W_vocab(d_model, vocab_size);

    std::vector<Tensor*> Wqkv(num_layers);
    std::vector<Tensor*> W1(num_layers);
    std::vector<Tensor*> b1(num_layers);
    std::vector<Tensor*> W2(num_layers);
    std::vector<Tensor*> b2(num_layers);

    for (int i = 0; i < num_layers; i++){
        Wqkv[i] = new Tensor(d_model, 3*d_model);
        W1[i] = new Tensor(d_model, d_ff);
        b1[i] = new Tensor(1, d_ff);
        W2[i] = new Tensor(d_ff, d_model);
        b2[i] = new Tensor(1, d_model);
        Wqkv[i]->fill(0.5f);
        W1[i]->fill(0.5f);
        b1[i]->fill(0.1f);
        W2[i]->fill(0.5f);
        b2[i]->fill(0.1f);
    }

    embedding.fill(0.5f);
    W_vocab.fill(0.5f);

    Tensor output(seq_len, vocab_size);

    transformer_forward(d_tokens, embedding, Wqkv.data(), W1.data(), b1.data(), W2.data(), b2.data(), W_vocab, output, num_layers, num_heads, seq_len);

    float* host_out1 = new float[seq_len * vocab_size];
    output.toCPU(host_out1);

    // Run again to ensure determinism
    transformer_forward(d_tokens, embedding, Wqkv.data(), W1.data(), b1.data(), W2.data(), b2.data(), W_vocab, output, num_layers, num_heads, seq_len);
    float* host_out2 = new float[seq_len * vocab_size];
    output.toCPU(host_out2);

    bool equal = arrays_close(host_out1, host_out2, seq_len*vocab_size, 1e-6f);
    bool finite = true;
    for(int i=0;i<seq_len*vocab_size;i++) if(!isfinite(host_out1[i])) finite=false;

    bool ok = equal && finite;
    report("test_transformer_smoke", ok);

    delete[] host_out1;
    delete[] host_out2;
    cudaFree(d_tokens);
    for (int i = 0; i < num_layers; i++){
        delete Wqkv[i]; delete W1[i]; delete b1[i]; delete W2[i]; delete b2[i];
    }
    return ok;
}
