using JTensors: CLBackend
ctx = CLBackend.init()
using OpenCL: cl
source = """
__kernel void reduce(
        __global float* buffer,
        __local float* scratch,
        __const int length,
        __global float* result
    ) {

    int global_index = get_global_id(0);
    float accumulator = INFINITY;
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float element = buffer[global_index];
        accumulator = (accumulator < element) ? accumulator : element;
        global_index += get_global_size(0);
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(
        int offset = get_local_size(0) / 2;
        offset > 0;
        offset = offset / 2
    ) {
        if (local_index < offset) {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = (mine < other) ? mine : other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}
"""
p = cl.build!(
    cl.Program(ctx.context, source = source),
    options = "-cl-denorms-are-zero -cl-mad-enable -cl-fast-relaxed-math"
)
k = cl.Kernel(p, "reduce")
blocks = 512
a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf = rand(Float32, 10^7))
b_buff = cl.Buffer(Float32, ctx, :w, blocks)
c_buff = cl.Buffer(Float32, ctx, :w, length(a))

p = cl.Program(ctx, source=sum_kernel) |> cl.build!
k = cl.Kernel(p, "sum")

queue(k, size(a), nothing, a_buff, b_buff, c_buff)

r = cl.read(queue, c_buff)
size_t globalWorkSize = 60 * 1024;
size_t localWorkSize = 128;
