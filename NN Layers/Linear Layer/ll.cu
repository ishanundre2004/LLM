#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define Batch 64
#define D_IN 512
#define D_OUT 1024




__global__ void add_bias(float* Y, const float* b, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        Y[idx] += b[col];
    }
}



int main(){

    size_t sizeX = Batch * D_IN * sizeof(float);
    size_t sizeW = D_IN * D_OUT * sizeof(float);
    size_t sizeY = Batch * D_OUT * sizeof(float);
    size_t sizeB = D_OUT * sizeof(float);

    float *h_X = (float*)malloc(sizeX);
    float *h_W = (float*)malloc(sizeW);
    float *h_Y = (float*)malloc(sizeY);
    float *h_B = (float*)malloc(sizeB);

    for(int i = 0; i < (Batch * D_IN); i++) h_X[i] = 1.0f;
    for(int i = 0; i < (D_IN * D_OUT); i++) h_W[i] = 0.01f;
    for(int i = 0; i < (D_OUT); i++) h_B[i] = 1.0f;

    float *d_X, *d_W, *d_Y, *d_B;

    cudaMalloc(&d_X, sizeX);
    cudaMalloc(&d_W, sizeW);
    cudaMalloc(&d_Y, sizeY);
    cudaMalloc(&d_B, sizeB);

    cudaMemcpy(d_X, h_X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        D_OUT, Batch, D_IN, 
        &alpha,
        d_W, D_OUT,
        d_X, D_IN,
        &beta, 
        d_Y, D_OUT
    );

    int threads = 256;
    int blocks = (Batch * D_OUT + threads - 1) / threads;
    add_bias<<<blocks, threads>>>(d_Y, d_B, Batch, D_OUT);

    cudaMemcpy(h_Y, d_Y, sizeY, cudaMemcpyDeviceToHost);

    printf("Y[0] = %f\n", h_Y[0]);

    cublasDestroy(handle);
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
    cudaFree(d_B);

    free(h_X);
    free(h_W);
    free(h_Y);
    free(h_B);



    return 0;
}