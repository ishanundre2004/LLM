#include <iostream>
#include "include/test_utils.h"
#include "../tensor/tensor.h"
#include "../attention/qkv/qkv.h"

bool test_matmul(){
    const int M=2,K=3,N=2;
    float A_h[M*K] = {1,2,3, 4,5,6}; // row-major 2x3
    float B_h[K*N] = {7,8, 9,10, 11,12}; // 3x2
    float C_ref[M*N];
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float s=0.0f;
            for(int k=0;k<K;k++) s += A_h[i*K+k]*B_h[k*N+j];
            C_ref[i*N+j]=s;
        }
    }

    Tensor A(M,K), B(K,N), C(M,N);
    A.toGPU(A_h);
    B.toGPU(B_h);

    matmul(A,B,C);

    float C_h[M*N];
    C.toCPU(C_h);

    bool ok = arrays_close(C_h, C_ref, M*N, 1e-3f);
    report("test_matmul", ok);
    return ok;
}
