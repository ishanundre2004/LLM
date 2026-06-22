#pragma once
#include <cmath>
#include <iostream>

inline bool approx_equal(float a, float b, float eps=1e-4f){
    if (std::isnan(a) && std::isnan(b)) return true;
    if (!std::isfinite(a) && !std::isfinite(b)) return std::signbit(a) == std::signbit(b);
    return fabsf(a-b) <= eps;
}

inline bool arrays_close(const float* a, const float* b, int n, float eps=1e-4f){
    for(int i=0;i<n;i++) if(!approx_equal(a[i], b[i], eps)) return false;
    return true;
}

inline void report(const char* name, bool ok){
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << std::endl;
}
