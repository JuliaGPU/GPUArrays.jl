using GPUArrays
using GPUArrays.CLBackend
CLBackend.init()


function clmap!(state, f, out, b)
    i = linear_index(out, state) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
    return
end
# we need a `guiding array`, to get the context and indicate on what size we
# want to execute the kernel! This kind of scheme might change in the future
x = GPUArray(rand(Float32, 100))
y = GPUArray(rand(Float32, 100))
# same here, x is just passed to supply a kernel size!
gpu_call(clmap!, x, (sin, x, y))
sin.(Array(y)) ≈ Array(x)

# you can also use a kernel source string, directly with OpenCL code.
# note, that you loose all features, like automatically including dependant functions
# and templating
source = """
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void julia(
    __global float2 *q,
    __global ushort *output,
    ushort const maxiter)
{
    int gid = get_global_id(0);
    float nreal = 0;
    float real  = q[gid].x;
    float imag  = q[gid].y;

    output[gid] = 0;

    for(int curiter = 0; curiter < maxiter; curiter++) {
        if (real*real + imag*imag > 4.0f) {
            output[gid] = curiter;
        }
        nreal = real*real - imag*imag + -0.5f;
        imag  = 2*real*imag + 0.75f;
        real = nreal;
    }
}
"""
w = 2048 * 2;
h = 2048 * 2;
q = [Complex64(r,i) for i=1:-(2.0/w):-1, r=-1.5:(3.0/h):1.5];
q_buff = GPUArray(q)
o_buff = similar(q_buff, UInt16)

# you need to pass the name of the kernel you'd like to compile
# It will as well need the "guiding array"
gpu_call((source, :julia), o_buff, (q_buff, o_buff, UInt16(200)))
# save out image!
using FileIO, Colors
x = Array(o_buff)
x ./= maximum(x)
save("test.jpg", Gray.(x))


using GPUArrays
using GPUArrays.CUBackend
using CUDAnative
CUBackend.init()

# CUDA works exactly the same!
x = GPUArray(rand(Float32, 100))
y = GPUArray(rand(Float32, 100))
# CUFunction instead of CLFunction
# Note, that we need to use the sin of CUDAnative.
# This necessity will hopefully be removed soon
# same here, x is just passed to supply a kernel size!
gpu_call(clmap!, x, (CUDAnative.sin, x, y))

yjl = Array(y)
sin.(yjl) ≈ Array(x)

source = """
__global__ void copy(const float *input, float *output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = input[i];
}
"""
gpu_call((source, :copy), x, (x, y))
Array(x) == Array(y)
