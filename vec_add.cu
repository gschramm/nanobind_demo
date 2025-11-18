// vec_add.cu
#include <cuda_runtime.h>
#include <stdexcept>
#include "vec_add.h"

__global__ void vec_add_kernel(const float *a, const float *b, float *c, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

extern "C" {

void add_vectors_cpu(const float *a, const float *b, float *c, std::size_t n) {
    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; ++i)
        c[i] = a[i] + b[i];
}

void add_vectors_cuda(const float *a, const float *b, float *c, std::size_t n) {
    int block = 256;
    int grid  = static_cast<int>((n + block - 1) / block);

    vec_add_kernel<<<grid, block>>>(a, b, c, n);

    // Make CUDA failures visible to Python
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("vec_add_kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ")
                                 + cudaGetErrorString(err));
    }
}

} // extern "C"
