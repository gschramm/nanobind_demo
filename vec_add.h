#ifndef VEC_ADD_H
#define VEC_ADD_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void add_vectors_cpu(const float *a, const float *b, float *c, std::size_t n);
void add_vectors_cuda(const float *a, const float *b, float *c, std::size_t n);

#ifdef __cplusplus
}
#endif

#endif // VEC_ADD_H

