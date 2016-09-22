using CUDAnative, CUDAdrv
const cu = CUDAnative
dev = CuDevice(0)
ctx = CuContext(dev)
@target ptx function mandelbulb{T}(x0::T,y0::T,z0::T, n, iter)
    n = T(8)
    x,y,z = x0,y0,z0
    for i=1:iter
        r = cu.sqrt(x*x + y*y + z*z)::T
        theta = cu.atan2(cu.sqrt(x*x + y*y) , z)::T
        phi = cu.atan2(y,x)::T
        rn = cu.pow(r, n)::T
        x1 = rn * cu.sin(theta*n) * cu.cos(phi*n) + x0
        y1 = rn * cu.sin(theta*n) * cu.sin(phi*n) + y0
        z1 = rn * cu.cos(theta*n) + z0
        (x1*x1 + y1*y1 + z1*z1) > n && return T(i)
        x,y,z = x1,y1,z1
    end
    T(iter)
end
@target ptx @generated function broadcast_index{T, N}(arg::AbstractArray{T,N}, shape, idx)
    idx = ntuple(i->:(ifelse(s[$i] < shape[$i], 1, idx[$i])), N)
    expr = quote
        s = size(arg)::NTuple{N, Int}
        @inbounds i = CartesianIndex{N}(($(idx...),)::NTuple{N, Int})
        @inbounds return arg[i]::T
    end
end
@target ptx function broadcast_index{T}(arg::T, shape, idx)
    arg::T
end
@target ptx function broadcast_kernel(A, f, arg_1, arg_2, arg_3, arg_4, arg_5)
    i = Int((blockIdx().x-1) * blockDim().x + threadIdx().x)
    @inbounds if i <= length(A)
        sz = size(A)
        idx = CartesianIndex(ind2sub(sz, Int(i)))
        A[idx] = f(
            broadcast_index(arg_1, sz, idx),
            broadcast_index(arg_2, sz, idx),
            broadcast_index(arg_3, sz, idx),
            broadcast_index(arg_4, sz, idx),
            broadcast_index(arg_5, sz, idx)
        )
    end
    nothing
end

n = 100
vol = CuArray(Array(Float32, n, n, n));
x  = linspace(-1f0, 1f0, n);
x1 = reshape(x, (n, 1, 1));
x2 = reshape(x, (1, n, 1));
x3 = reshape(x, (1, 1, n));
len = length(vol)
threads = min(len, 1024)
blocks = ceil(Int, len/threads)
call_cuda(broadcast_kernel, vol, mandelbulb, x1, x2, x3, 8f0, 10)

#vol .= mandelbulb.(GPUArray(x1), GPUArray(x2), GPUArray(x3), 8f0, 10)



using CUDAdrv, CUDAnative
@target ptx function kernel{T}(A, v::T)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds A[i] = v[div(i, 10000), 1, 1]
    end
    nothing
end

dev = CuDevice(0)
ctx = CuContext(dev)
n = 100
x  = linspace(-1f0, 1f0, n);
xr = reshape(x, (n, 1, 1));
d_arr = CuArray(zeros(Float32, (n, n, n)));
len = length(d_arr)
threads = min(len, 1024)
blocks = ceil(Int, len/threads)
@cuda (threads, blocks) kernel(d_arr, xr)


using GPUArrays, CUDAnative
import GPUArrays: GPUArray, CUBackend, cu_map
cuctx = CUBackend.init()
const cu = CUDAnative
n = 10
vola = zeros(Float32, n, n, n)
vol = GPUArray(vola);
x1 = reshape(linspace(-1f0, 1f0, n), (n, 1, 1));
vol .= identity.(x1)
