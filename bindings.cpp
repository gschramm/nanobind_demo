// vecadd_ext.cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <vec_add.h>

#include <iostream>

namespace nb = nanobind;
// using nb::arg;
using namespace nb::literals;

// Accept *any* contiguous array from any framework / device.
// We'll enforce float32 + shape + device in C++.
using ConstFloatArray = nb::ndarray<const float, nb::c_contig>;
using FloatArray = nb::ndarray<float, nb::c_contig>;

void add_inplace_any(ConstFloatArray a, ConstFloatArray b, FloatArray out)
{
    // 2. Same number of dimensions
    if (a.ndim() != b.ndim() || a.ndim() != out.ndim())
    {
        throw nb::value_error("add_inplace: arrays must have same ndim");
    }

    // 3. Same shape everywhere
    std::size_t ndim = a.ndim();
    std::size_t n = 1;
    for (std::size_t i = 0; i < ndim; ++i)
    {
        auto sa = a.shape(i);
        if (sa != b.shape(i) || sa != out.shape(i))
        {
            throw nb::value_error("add_inplace: all inputs must have identical shapes");
        }
        n *= static_cast<std::size_t>(sa);
    }

    // 4. (Optional) your extra constraint: last dimension = 3
    if (a.shape(ndim - 1) != 3)
    {
        throw nb::value_error(
            "add_inplace: last dimension must be 3 (e.g. [..., 3])");
    }

    // calculate the product of all dimensions
    size_t nvoxels = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        nvoxels *= a.shape(i);
    }
    std::cout << "Total number of elements: " << nvoxels << "\n";

    int img_dim[3] = {static_cast<int>(a.shape(0)),
                      static_cast<int>(a.shape(1)),
                      static_cast<int>(a.shape(2))};

    // 5. Ensure all arrays live on the same device
    int dev_type = a.device_type();
    int dev_id = a.device_id();

    std::cout << "Device type: " << dev_type << ", Device ID: " << dev_id << "\n";

    if (b.device_type() != dev_type || out.device_type() != dev_type ||
        b.device_id() != dev_id || out.device_id() != dev_id)
    {
        throw nb::value_error("add_inplace: all arrays must be on the same device");
    }

    // 6. check that device is either cpu, cuda, cuda_host or cuda_managed
    if (dev_type != nb::device::cpu::value &&
        dev_type != nb::device::cuda::value &&
        dev_type != nb::device::cuda_managed::value &&
        dev_type != nb::device::cuda_host::value)
    {
        throw nb::value_error("add_inplace: unsupported device type");
    }

    // 7. Call the appropriate backend
    if (dev_type == nb::device::cpu::value)
    {
        // NumPy, torch CPU, jax CPU, ...
        add_vectors_cpu(a.data(), b.data(), out.data(), n);
    }
    else if (dev_type == nb::device::cuda::value)
    {
        // CuPy, torch.cuda, jax GPU, ...
        add_vectors_cuda(a.data(), b.data(), out.data(), n);
    }
    else
    {
        throw nb::value_error("add_inplace: unsupported device type");
    }
}

NB_MODULE(vecadd_ext, m)
{
    m.def("add_inplace",
          &add_inplace_any,
          "a"_a.noconvert(), "b"_a.noconvert(), "out"_a.noconvert(),
          "In-place elementwise add: out = a + b for float32 arrays.\n"
          "Supports NumPy / CuPy / PyTorch CPU and CUDA tensors.\n"
          "Shape must match and last dimension must be 3.");
}
