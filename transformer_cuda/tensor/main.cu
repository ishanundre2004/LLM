#include "tensor.h"

int main(){
    Tensor A(4, 4);
    A.fill(1.0f);
    A.print();
    return 0;
}