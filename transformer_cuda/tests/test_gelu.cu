#include <cmath>
#include <iostream>
#include <vector>
#include "include/test_utils.h"
#include "../tensor/tensor.h"
#include "../kernels/gelu/gelu.h"

inline float gelu_cpu(float x){
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

bool test_gelu(){
    const int R = 2;
    const int C = 4;
    const int size = R * C;
    std::vector<float> in_h(size);
    for(int i=0;i<size;i++) in_h[i] = (float)(i-3);

    Tensor T(R,C);
    T.toGPU(in_h.data());

    gelu(T);

    std::vector<float> out_h(size);
    T.toCPU(out_h.data());

    std::vector<float> ref(size);
    for(int i=0;i<size;i++) ref[i] = gelu_cpu(in_h[i]);

    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-3f);
    report("test_gelu", ok);
    return ok;
}
