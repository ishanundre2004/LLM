#include <cmath>
#include <iostream>
#include <vector>
#include "include/test_utils.h"
#include "../tensor/tensor.h"
#include "../attention/multihead/multihead.h"
#include "../kernels/softmax/softmax.h"  // Include softmax header

// CPU reference for simple multi-head attention (no masking)
bool cpu_mha_ref(const float* Q, const float* K, const float* V,
                 float* O, int seq_len, int d_model, int num_heads)
{
    int d_k = d_model / num_heads;
    float scale = 1.0f / sqrtf((float)d_k);
    
    for(int row=0; row<seq_len; ++row){
        for(int h=0; h<num_heads; ++h){
            int head_off = h*d_k;
            // compute attention weights
            float scores[16];  // Note: hardcoded to max 16, consider dynamic allocation
            float max_score = -INFINITY;
            
            // Step 1: Compute scores and find max (numerically stable)
            for(int j=0; j<seq_len; j++){
                float dot = 0.0f;
                for(int k=0; k<d_k; k++){
                    dot += Q[row*d_model + head_off + k] * K[j*d_model + head_off + k];
                }
                scores[j] = dot * scale;
                max_score = fmaxf(max_score, scores[j]);
            }
            
            // Step 2: Softmax with numerical stability
            float sumexp = 0.0f;
            for(int j=0; j<seq_len; j++){ 
                scores[j] = expf(scores[j] - max_score); 
                sumexp += scores[j]; 
            }
            
            // Step 3: Weighted sum
            for(int k=0; k<d_k; k++){
                float val = 0.0f;
                for(int j=0; j<seq_len; j++){
                    val += (scores[j] / sumexp) * V[j*d_model + head_off + k];
                }
                O[row*d_model + head_off + k] = val;
            }
        }
    }
    return true;
}

// CPU reference for softmax (numerically stable)
void cpu_softmax_ref(const float* input, float* output, int rows, int cols, float scale = 1.0f) {
    for (int r = 0; r < rows; ++r) {
        // Find max for stability
        float max_val = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            float val = input[r * cols + c] * scale;
            max_val = fmaxf(max_val, val);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        std::vector<float> temp(cols);
        for (int c = 0; c < cols; ++c) {
            float val = expf(input[r * cols + c] * scale - max_val);
            temp[c] = val;
            sum += val;
        }
        
        // Normalize
        for (int c = 0; c < cols; ++c) {
            output[r * cols + c] = temp[c] / sum;
        }
    }
}

// Test multi-head attention
bool test_attention() {
    const int seq_len = 2;
    const int d_model = 4;
    const int num_heads = 2;
    const int size = seq_len * d_model;
    
    std::vector<float> Q_h(size);
    std::vector<float> K_h(size);
    std::vector<float> V_h(size);
    
    for(int i=0; i<size; i++) { 
        Q_h[i] = (float)(i+1); 
        K_h[i] = (float)(i+2); 
        V_h[i] = (float)(i+3); 
    }

    Tensor Q(seq_len, d_model), K(seq_len, d_model), V(seq_len, d_model), O(seq_len, d_model);
    Q.toGPU(Q_h.data()); 
    K.toGPU(K_h.data()); 
    V.toGPU(V_h.data());

    multihead_attention(Q, K, V, O, num_heads);

    std::vector<float> out_h(size);
    O.toCPU(out_h.data());

    std::vector<float> ref(size);
    cpu_mha_ref(Q_h.data(), K_h.data(), V_h.data(), ref.data(), seq_len, d_model, num_heads);

    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-3f);
    
    if(!ok) {
        std::cout << "GPU output:\n";
        for(int i=0; i<size; i++) std::cout << out_h[i] << " ";
        std::cout << "\nCPU reference:\n";
        for(int i=0; i<size; i++) std::cout << ref[i] << " ";
        std::cout << "\nDiffs:\n";
        for(int i=0; i<size; i++) std::cout << (out_h[i]-ref[i]) << " ";
        std::cout << std::endl;
    }

    report("test_attention", ok);
    return ok;
}

// Test softmax - normal case
bool test_softmax_normal() {
    const int rows = 4;
    const int cols = 8;
    const int size = rows * cols;
    
    std::vector<float> input_h(size);
    
    // Generate random-ish input
    for(int i=0; i<size; i++) {
        input_h[i] = sinf((float)i) * 2.0f;
    }
    
    Tensor input(rows, cols), output(rows, cols);
    input.toGPU(input_h.data());
    
    softmax(input, output, rows, cols);
    
    std::vector<float> out_h(size);
    output.toCPU(out_h.data());
    
    std::vector<float> ref(size);
    cpu_softmax_ref(input_h.data(), ref.data(), rows, cols);
    
    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-5f);
    
    if(!ok) {
        std::cout << "Max diff: ";
        float max_diff = 0.0f;
        for(int i=0; i<size; i++) {
            max_diff = fmaxf(max_diff, fabsf(out_h[i] - ref[i]));
        }
        std::cout << max_diff << std::endl;
        
        // Print first row for debugging
        std::cout << "First row GPU: ";
        for(int i=0; i<cols; i++) std::cout << out_h[i] << " ";
        std::cout << "\nFirst row CPU: ";
        for(int i=0; i<cols; i++) std::cout << ref[i] << " ";
        std::cout << std::endl;
    }
    
    report("test_softmax_normal", ok);
    return ok;
}

// Test softmax - extreme values
bool test_softmax_extreme() {
    const int rows = 1;
    const int cols = 3;
    const int size = rows * cols;
    
    std::vector<float> input_h = {1000.0f, 1001.0f, 1002.0f};
    
    Tensor input(rows, cols), output(rows, cols);
    input.toGPU(input_h.data());
    
    softmax(input, output, rows, cols);
    
    std::vector<float> out_h(size);
    output.toCPU(out_h.data());
    
    std::vector<float> ref(size);
    cpu_softmax_ref(input_h.data(), ref.data(), rows, cols);
    
    // Use slightly relaxed tolerance for extreme values
    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-4f);
    
    if(!ok) {
        std::cout << "Input: ";
        for(int i=0; i<cols; i++) std::cout << input_h[i] << " ";
        std::cout << "\nGPU: ";
        for(int i=0; i<cols; i++) std::cout << out_h[i] << " ";
        std::cout << "\nCPU: ";
        for(int i=0; i<cols; i++) std::cout << ref[i] << " ";
        std::cout << std::endl;
        
        // Check if any NaN or Inf
        bool has_nan = false;
        for(int i=0; i<size; i++) {
            if(isnan(out_h[i]) || isinf(out_h[i])) {
                has_nan = true;
                break;
            }
        }
        if(has_nan) {
            std::cout << "WARNING: GPU output contains NaN or Inf!" << std::endl;
        }
    }
    
    report("test_softmax_extreme", ok);
    return ok;
}

// Test softmax - scaled version (attention-style)
bool test_softmax_scaled() {
    const int rows = 2;
    const int cols = 8;
    const int size = rows * cols;
    float scale = 1.0f / sqrtf((float)cols);  // d_k = 8
    
    std::vector<float> input_h(size);
    
    for(int i=0; i<size; i++) {
        input_h[i] = cosf((float)i) * 3.0f;
    }
    
    Tensor input(rows, cols), output(rows, cols);
    input.toGPU(input_h.data());
    
    softmax_scaled(input, output, scale);
    
    std::vector<float> out_h(size);
    output.toCPU(out_h.data());
    
    std::vector<float> ref(size);
    cpu_softmax_ref(input_h.data(), ref.data(), rows, cols, scale);
    
    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-5f);
    
    if(!ok) {
        float max_diff = 0.0f;
        for(int i=0; i<size; i++) {
            max_diff = fmaxf(max_diff, fabsf(out_h[i] - ref[i]));
        }
        std::cout << "Max diff with scale=" << scale << ": " << max_diff << std::endl;
    }
    
    report("test_softmax_scaled", ok);
    return ok;
}

// Test softmax - large batch
bool test_softmax_large() {
    const int rows = 256;
    const int cols = 128;
    const int size = rows * cols;
    
    std::vector<float> input_h(size);
    
    // Random values
    srand(42);
    for(int i=0; i<size; i++) {
        input_h[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
    }
    
    Tensor input(rows, cols), output(rows, cols);
    input.toGPU(input_h.data());
    
    softmax(input, output, rows, cols);
    
    // Only check first and last rows for speed
    std::vector<float> out_h(size);
    output.toCPU(out_h.data());
    
    std::vector<float> ref(size);
    cpu_softmax_ref(input_h.data(), ref.data(), rows, cols);
    
    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-5f);
    
    if(!ok) {
        float max_diff = 0.0f;
        for(int i=0; i<size; i++) {
            max_diff = fmaxf(max_diff, fabsf(out_h[i] - ref[i]));
        }
        std::cout << "Max diff (large): " << max_diff << std::endl;
    }
    
    report("test_softmax_large", ok);
    return ok;
}

// Test softmax - in-place version
bool test_softmax_inplace() {
    const int rows = 4;
    const int cols = 16;
    const int size = rows * cols;
    
    std::vector<float> input_h(size);
    
    for(int i=0; i<size; i++) {
        input_h[i] = ((float)(i % 17) - 8.0f);
    }
    
    Tensor data(rows, cols);
    data.toGPU(input_h.data());
    
    // In-place softmax
    softmax_inplace(data, 1.0f);
    
    std::vector<float> out_h(size);
    data.toCPU(out_h.data());
    
    std::vector<float> ref(size);
    cpu_softmax_ref(input_h.data(), ref.data(), rows, cols);
    
    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-5f);
    
    if(!ok) {
        float max_diff = 0.0f;
        for(int i=0; i<size; i++) {
            max_diff = fmaxf(max_diff, fabsf(out_h[i] - ref[i]));
        }
        std::cout << "Max diff (inplace): " << max_diff << std::endl;
    }
    
    report("test_softmax_inplace", ok);
    return ok;
}

// Main test runner
int main() {
    std::cout << "=== Running GPU Kernel Tests ===\n" << std::endl;
    
    bool all_passed = true;
    
    // Run attention test
    std::cout << "Testing Multi-Head Attention..." << std::endl;
    all_passed &= test_attention();
    
    // Run softmax tests
    std::cout << "\nTesting Softmax Kernels..." << std::endl;
    all_passed &= test_softmax_normal();
    all_passed &= test_softmax_extreme();
    all_passed &= test_softmax_scaled();
    all_passed &= test_softmax_large();
    all_passed &= test_softmax_inplace();
    
    std::cout << "\n=== All Tests " << (all_passed ? "PASSED" : "FAILED") << " ===" << std::endl;
    
    return all_passed ? 0 : 1;
}