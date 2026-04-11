#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

#include "tensor/tensor.h"
#include "model/transformer/transformer.h"

// ----------------------
// Simple vocab
// ----------------------
std::vector<std::string> vocab = {
    "I", "love", "AI", "CUDA", "GPU", "is", "awesome"
};

std::unordered_map<std::string, int> token_to_id = {
    {"I",0}, {"love",1}, {"AI",2}, {"CUDA",3},
    {"GPU",4}, {"is",5}, {"awesome",6}
};

// ----------------------
// Argmax helper (host)
// ----------------------
int argmax(float* logits, int size) {
    float max_val = logits[0];
    int idx = 0;
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            idx = i;
        }
    }
    return idx;
}

// ----------------------
// Main
// ----------------------
int main() {
    int seq_len = 3;
    int vocab_size = vocab.size();
    int d_model = 8;
    int d_ff = 32;
    int num_heads = 2;
    int num_layers = 1;

    std::cout << "Mini Transformer Demo (CUDA)\n";

    // ----------------------
    // Input tokens
    // ----------------------
    std::vector<int> tokens_host = {
        token_to_id["I"],
        token_to_id["love"],
        token_to_id["AI"]
    };

    int* tokens_device = nullptr;
    cudaMalloc(&tokens_device, seq_len * sizeof(int));
    cudaMemcpy(tokens_device, tokens_host.data(),
               seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // ----------------------
    // Model parameters
    // ----------------------
    Tensor embedding(vocab_size, d_model);
    Tensor W_vocab(d_model, vocab_size);

    std::vector<Tensor*> Wqkv(num_layers);
    std::vector<Tensor*> W1(num_layers);
    std::vector<Tensor*> b1(num_layers);
    std::vector<Tensor*> W2(num_layers);
    std::vector<Tensor*> b2(num_layers);

    for (int i = 0; i < num_layers; i++) {
        Wqkv[i] = new Tensor(d_model, 3 * d_model);
        W1[i]   = new Tensor(d_model, d_ff);
        b1[i]   = new Tensor(1, d_ff);
        W2[i]   = new Tensor(d_ff, d_model);
        b2[i]   = new Tensor(1, d_model);
    }

    // ----------------------
    // Initialize all weights & biases
    // ----------------------
    embedding.fill(0.5f);
    W_vocab.fill(0.5f);

    for (int i = 0; i < num_layers; i++) {
        Wqkv[i]->fill(0.5f);
        W1[i]->fill(0.5f);
        b1[i]->fill(0.1f);
        W2[i]->fill(0.5f);
        b2[i]->fill(0.1f);
    }

    // ----------------------
    // Output tensor (logits)
    // ----------------------
    Tensor output(seq_len, vocab_size);

    // ----------------------
    // Timing events
    // ----------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ----------------------
    // Transformer forward pass
    // ----------------------
    transformer_forward(
        tokens_device,
        embedding,
        Wqkv.data(),
        W1.data(), b1.data(),
        W2.data(), b2.data(),
        W_vocab,
        output,
        num_layers,
        num_heads,
        seq_len
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // ----------------------
    // Copy logits back to host
    // ----------------------
    float* host_out = new float[seq_len * vocab_size];
    output.toCPU(host_out);

    // ----------------------
    // Predict next token from last position
    // ----------------------
    int last_row = seq_len - 1;
    int predicted = argmax(&host_out[last_row * vocab_size], vocab_size);

    // ----------------------
    // Print results
    // ----------------------
    std::cout << "\nInput: ";
    for (int t : tokens_host)
        std::cout << vocab[t] << " ";
    std::cout << "\nPredicted: " << vocab[predicted] << std::endl;
    std::cout << "Time: " << ms << " ms\n";

    // ----------------------
    // Cleanup
    // ----------------------
    cudaFree(tokens_device);
    delete[] host_out;

    for (int i = 0; i < num_layers; i++) {
        delete Wqkv[i];
        delete W1[i];
        delete b1[i];
        delete W2[i];
        delete b2[i];
    }

    return 0;
}