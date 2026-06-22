#include <cmath>
#include <iostream>
#include <vector>
#include <cfloat>
#include "include/test_utils.h"
#include "../tensor/tensor.h"
#include "../attention/multihead/multihead.h"
#include "../kernels/softmax/softmax.h"
#include "../kernels/gelu/gelu.h"

// ============================================================
// CPU REFERENCE FUNCTIONS
// ============================================================

// GELU CPU reference
inline float gelu_cpu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// CPU softmax reference
void cpu_softmax_ref(const float* input, float* output, int rows, int cols, float scale = 1.0f) {
    for (int r = 0; r < rows; r++) {
        float max_val = -FLT_MAX;
        for (int c = 0; c < cols; c++) {
            max_val = fmaxf(max_val, input[r*cols+c] * scale);
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float val = expf(input[r*cols+c] * scale - max_val);
            output[r*cols+c] = val;
            sum += val;
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            output[r*cols+c] *= inv_sum;
        }
    }
}

// CPU attention reference (no mask)
void cpu_attention_ref(const float* Q, const float* K, const float* V, float* O,
                       int seq_len, int d_model, int num_heads) {
    int d_k = d_model / num_heads;
    float scale = 1.0f / sqrtf((float)d_k);
    
    for (int row = 0; row < seq_len; row++) {
        for (int h = 0; h < num_heads; h++) {
            int head_off = h * d_k;
            std::vector<float> scores(seq_len);
            float max_score = -FLT_MAX;
            
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int k = 0; k < d_k; k++) {
                    dot += Q[row*d_model + head_off + k] * K[j*d_model + head_off + k];
                }
                scores[j] = dot * scale;
                max_score = fmaxf(max_score, scores[j]);
            }
            
            float sumexp = 0.0f;
            for (int j = 0; j < seq_len; j++) { 
                scores[j] = expf(scores[j] - max_score); 
                sumexp += scores[j]; 
            }
            
            for (int k = 0; k < d_k; k++) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    val += (scores[j] / sumexp) * V[j*d_model + head_off + k];
                }
                O[row*d_model + head_off + k] = val;
            }
        }
    }
}

// CPU attention reference (causal mask)
void cpu_attention_causal_ref(const float* Q, const float* K, const float* V, float* O,
                               int seq_len, int d_model, int num_heads) {
    int d_k = d_model / num_heads;
    float scale = 1.0f / sqrtf((float)d_k);
    
    for (int row = 0; row < seq_len; row++) {
        for (int h = 0; h < num_heads; h++) {
            int head_off = h * d_k;
            std::vector<float> scores(seq_len);
            float max_score = -FLT_MAX;
            
            // CAUSAL: Only attend to positions j <= row
            for (int j = 0; j <= row; j++) {
                float dot = 0.0f;
                for (int k = 0; k < d_k; k++) {
                    dot += Q[row*d_model + head_off + k] * K[j*d_model + head_off + k];
                }
                scores[j] = dot * scale;
                max_score = fmaxf(max_score, scores[j]);
            }
            
            // Set future positions to -inf
            for (int j = row + 1; j < seq_len; j++) {
                scores[j] = -INFINITY;
            }
            
            float sumexp = 0.0f;
            for (int j = 0; j < seq_len; j++) { 
                if (scores[j] == -INFINITY) {
                    scores[j] = 0.0f;
                } else {
                    scores[j] = expf(scores[j] - max_score);
                }
                sumexp += scores[j]; 
            }
            
            for (int k = 0; k < d_k; k++) {
                float val = 0.0f;
                for (int j = 0; j <= row; j++) {
                    if (scores[j] > 0.0f) {
                        val += (scores[j] / sumexp) * V[j*d_model + head_off + k];
                    }
                }
                O[row*d_model + head_off + k] = val;
            }
        }
    }
}

// ============================================================
// EXISTING TESTS (UNCHANGED)
// ============================================================

// Test 1: GELU
bool test_gelu() {
    const int R = 2;
    const int C = 4;
    const int size = R * C;
    std::vector<float> in_h(size);
    for (int i = 0; i < size; i++) in_h[i] = (float)(i-3);

    Tensor T(R, C);
    T.toGPU(in_h.data());
    gelu(T);

    std::vector<float> out_h(size);
    T.toCPU(out_h.data());

    std::vector<float> ref(size);
    for (int i = 0; i < size; i++) ref[i] = gelu_cpu(in_h[i]);

    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-3f);
    report("test_gelu", ok);
    return ok;
}

// Test 2: Softmax Normal
bool test_softmax_normal() {
    const int rows = 4, cols = 8, size = rows * cols;
    std::vector<float> input(size), gpu_out(size), cpu_out(size);
    for (int i = 0; i < size; i++) input[i] = sinf((float)i);

    Tensor d_input(rows, cols), d_output(rows, cols);
    d_input.toGPU(input.data());
    softmax(d_input, d_output, rows, cols);
    d_output.toCPU(gpu_out.data());
    cpu_softmax_ref(input.data(), cpu_out.data(), rows, cols);

    bool ok = arrays_close(gpu_out.data(), cpu_out.data(), size, 1e-5f);
    report("test_softmax_normal", ok);
    return ok;
}

// Test 3: Softmax Extreme Values
bool test_softmax_extreme() {
    const int rows = 1, cols = 3, size = cols;
    std::vector<float> input = {1000.0f, 1001.0f, 1002.0f};
    std::vector<float> gpu_out(size), cpu_out(size);

    Tensor d_input(rows, cols), d_output(rows, cols);
    d_input.toGPU(input.data());
    softmax(d_input, d_output, rows, cols);
    d_output.toCPU(gpu_out.data());
    cpu_softmax_ref(input.data(), cpu_out.data(), rows, cols);

    bool ok = arrays_close(gpu_out.data(), cpu_out.data(), size, 1e-4f);
    report("test_softmax_extreme", ok);
    return ok;
}

// Test 4: Softmax Scaled
bool test_softmax_scaled() {
    const int rows = 2, cols = 8, size = rows * cols;
    float scale = 1.0f / sqrtf((float)cols);
    std::vector<float> input(size), gpu_out(size), cpu_out(size);
    for (int i = 0; i < size; i++) input[i] = cosf((float)i) * 3.0f;

    Tensor d_input(rows, cols), d_output(rows, cols);
    d_input.toGPU(input.data());
    softmax_scaled(d_input, d_output, scale);
    d_output.toCPU(gpu_out.data());
    cpu_softmax_ref(input.data(), cpu_out.data(), rows, cols, scale);

    bool ok = arrays_close(gpu_out.data(), cpu_out.data(), size, 1e-5f);
    report("test_softmax_scaled", ok);
    return ok;
}

// Test 5: Attention (no mask)
bool test_attention() {
    const int seq_len = 2;
    const int d_model = 4;
    const int num_heads = 2;
    const int size = seq_len * d_model;
    
    std::vector<float> Q_h(size), K_h(size), V_h(size);
    for (int i = 0; i < size; i++) { 
        Q_h[i] = (float)(i+1); 
        K_h[i] = (float)(i+2); 
        V_h[i] = (float)(i+3); 
    }

    Tensor Q(seq_len, d_model), K(seq_len, d_model), V(seq_len, d_model), O(seq_len, d_model);
    Q.toGPU(Q_h.data()); 
    K.toGPU(K_h.data()); 
    V.toGPU(V_h.data());

    multihead_attention(Q, K, V, O, num_heads);

    std::vector<float> out_h(size), ref(size);
    O.toCPU(out_h.data());
    cpu_attention_ref(Q_h.data(), K_h.data(), V_h.data(), ref.data(), seq_len, d_model, num_heads);

    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-3f);
    report("test_attention", ok);
    return ok;
}

// ============================================================
// NEW TESTS FOR CAUSAL ATTENTION
// ============================================================

// Test 6: Causal Attention
bool test_attention_causal() {
    const int seq_len = 3;  // Use 3 to clearly verify triangular mask
    const int d_model = 4;
    const int num_heads = 2;
    const int size = seq_len * d_model;
    
    std::vector<float> Q_h(size), K_h(size), V_h(size);
    for (int i = 0; i < size; i++) { 
        Q_h[i] = (float)(i+1); 
        K_h[i] = (float)(i+2); 
        V_h[i] = (float)(i+3); 
    }

    Tensor Q(seq_len, d_model), K(seq_len, d_model), V(seq_len, d_model), O(seq_len, d_model);
    Q.toGPU(Q_h.data()); 
    K.toGPU(K_h.data()); 
    V.toGPU(V_h.data());

    multihead_attention_causal(Q, K, V, O, num_heads);

    std::vector<float> out_h(size), ref(size);
    O.toCPU(out_h.data());
    cpu_attention_causal_ref(Q_h.data(), K_h.data(), V_h.data(), ref.data(), seq_len, d_model, num_heads);

    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-3f);
    report("test_attention_causal", ok);
    return ok;
}

// Test 7: Causal Mask Properties
bool test_attention_causal_properties() {
    // Verify that causal mask prevents attending to future tokens
    const int seq_len = 4;
    const int d_model = 4;
    const int num_heads = 2;
    const int size = seq_len * d_model;
    
    // Create inputs where K[0] and V[0] have large values
    // If causal is wrong, position 2 might attend to position 3
    std::vector<float> Q_h(size, 0.0f);
    std::vector<float> K_h(size, 0.0f);
    std::vector<float> V_h(size, 0.0f);
    
    // Make token 0 have extreme values
    for (int i = 0; i < d_model; i++) {
        K_h[0 * d_model + i] = 1000.0f;  // Very large K for token 0
        V_h[0 * d_model + i] = 100.0f;   // Very large V for token 0
    }
    // Make token 3 have extreme values
    for (int i = 0; i < d_model; i++) {
        K_h[3 * d_model + i] = 1000.0f;  // Very large K for token 3
        V_h[3 * d_model + i] = 999.0f;   // Different V to distinguish
    }
    
    // Small random Q values
    for (int i = 0; i < size; i++) {
        Q_h[i] = 1.0f;
    }

    Tensor Q(seq_len, d_model), K(seq_len, d_model), V(seq_len, d_model), O(seq_len, d_model);
    Q.toGPU(Q_h.data()); 
    K.toGPU(K_h.data()); 
    V.toGPU(V_h.data());

    multihead_attention_causal(Q, K, V, O, num_heads);

    std::vector<float> out_h(size);
    O.toCPU(out_h.data());
    
    // For position 0: should only see token 0 (V=100), not token 3 (V=999)
    // For position 1: should see tokens 0,1 (mostly token 0 with V=100)
    // For position 2: should not see token 3 (V=999) due to causal mask
    
    bool causal_correct = true;
    // Position 2's output should be close to 100 (from token 0) not 999 (from token 3)
    for (int i = 0; i < d_model; i++) {
        float val = out_h[2 * d_model + i];
        if (fabsf(val - 999.0f) < fabsf(val - 100.0f)) {
            // Value is closer to 999 than 100, meaning it saw future token
            causal_correct = false;
            break;
        }
    }
    
    bool ok = arrays_close(out_h.data(), out_h.data(), size, 0.0f) || causal_correct;
    report("test_attention_causal_properties", causal_correct);
    return causal_correct;
}

// Test 8: Masked Attention (Custom Mask)
bool test_attention_masked() {
    const int seq_len = 3;
    const int d_model = 4;
    const int num_heads = 2;
    const int size = seq_len * d_model;
    
    std::vector<float> Q_h(size), K_h(size), V_h(size);
    for (int i = 0; i < size; i++) { 
        Q_h[i] = (float)(i+1); 
        K_h[i] = (float)(i+2); 
        V_h[i] = (float)(i+3); 
    }
    
    // Create custom mask: allow all positions
    std::vector<float> mask_h(seq_len * seq_len, 1.0f);
    // Mask out position 1 -> position 2 specifically
    mask_h[1 * seq_len + 2] = 0.0f;

    Tensor Q(seq_len, d_model), K(seq_len, d_model), V(seq_len, d_model), O(seq_len, d_model);
    Tensor mask(seq_len, seq_len);
    
    Q.toGPU(Q_h.data()); 
    K.toGPU(K_h.data()); 
    V.toGPU(V_h.data());
    mask.toGPU(mask_h.data());

    multihead_attention_masked(Q, K, V, mask, O, num_heads);

    std::vector<float> out_h(size);
    O.toCPU(out_h.data());
    
    // Basic sanity check: output should be valid (not NaN)
    bool ok = true;
    for (int i = 0; i < size; i++) {
        if (isnan(out_h[i]) || isinf(out_h[i])) {
            ok = false;
            break;
        }
    }
    
    report("test_attention_masked", ok);
    return ok;
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  CUDA Transformer Kernel Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    bool all_passed = true;
    
    std::cout << "Activation Function Tests:" << std::endl;
    all_passed &= test_gelu();
    
    std::cout << "\nSoftmax Tests:" << std::endl;
    all_passed &= test_softmax_normal();
    all_passed &= test_softmax_extreme();
    all_passed &= test_softmax_scaled();
    
    std::cout << "\nAttention Tests:" << std::endl;
    all_passed &= test_attention();
    
    std::cout << "\nCausal & Masked Attention Tests:" << std::endl;
    all_passed &= test_attention_causal();
    all_passed &= test_attention_causal_properties();
    all_passed &= test_attention_masked();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return all_passed ? 0 : 1;
}