#include <cmath>
#include <iostream>
#include <vector>
#include "include/test_utils.h"
#include "../tensor/tensor.h"
#include "../layers/layernorm/layernorm.h"

bool test_layernorm(){
    const int R = 2;
    const int C = 4;
    const int size = R * C;
    std::vector<float> in_h = {1,2,3,4, 5,6,7,8};

    Tensor In(R,C), Out(R,C);
    In.toGPU(in_h.data());

    layernorm(In, Out);

    std::vector<float> out_h(size);
    Out.toCPU(out_h.data());

    // CPU reference per-row
    std::vector<float> ref(size);
    for(int r=0;r<R;r++){
        float sum=0.0f;
        for(int c=0;c<C;c++) sum += in_h[r*C+c];
        float mean = sum / C;
        float var=0.0f;
        for(int c=0;c<C;c++){ float d=in_h[r*C+c]-mean; var += d*d; }
        var /= C;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for(int c=0;c<C;c++) ref[r*C+c] = (in_h[r*C+c]-mean) * inv_std;
    }

    bool ok = arrays_close(out_h.data(), ref.data(), size, 1e-3f);
    report("test_layernorm", ok);
    return ok;
}
