#include <cuda_runtime.h>
#include <float.h>
#include "softmax.h"

__global__ void softmax_kernel(float* input, float* output, int rows int cols){
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) return;

    float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    
    float max_val = -FLT_MAX;

    for(int i = tid ; i < cols; i += blockDim.x){
        max_val = fmaxf(max_val, row_ptr[i]);
    }

    shared[tid] = max_val;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>= 2){
        if(tid < stride){
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();


    //for sum;
    float sum = 0.0f;

    for(int i = tid ;i < col ; i += blockDim.x){
        float val =  expf(row_ptr[i] - max_val);
        out_ptr[i] = val;
        sum += val;
    }

    shared[tid] = sum;
    __syncthreads();

    for(int stride = blockDim/2; stride > 0; stride >>= 1){
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();
    for(int i = tid; i < cols; i += blockDim.x){
        out_ptr[i] /= shared[0];
    }

}