#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

void embedding(Tensor& embeddings, int* token_ids, Tensor& output, int seq_len);

#endif