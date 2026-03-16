#include "tensor/tensor.h"
#include "kernels/softmax/softmax.h"

int main(){
    Tensor A(2,4);
    Tensor B(2,4);

    float host[] = {
        1,2,3,4,
        2,4,6,8
    };

    A.toGPU(host);

    softmax(A,B);   

    B.print();
    return 0;
}