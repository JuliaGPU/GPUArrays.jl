using Colors, ColorVectorSpace, FileIO
using GPUArrays
import GPUArrays: mapidx, AbstractAccArray

@inline function euclidian(a::Colorant, b::Colorant)
    abs(comp1(a) - comp1(b)) +
    abs(comp2(a) - comp2(b)) +
    abs(comp3(a) - comp3(b))
end
function labeldata{T}(val, means, k::T)
    label = T(0); mindist = Float32(Inf)
    @inbounds for ki = T(1):k
        dist = euclidian(means[ki], val)
        if dist < mindist
            mindist = dist
            label = ki
        end
    end
    return UInt8(label)
end

function mymap!(f, out, A, means, k, t)
    for i=1:length(A)
        out[i] = labeldata(A[i], means, k, T)
    end
    out
end
@inline function meancount{T}(v::T, label, k)
    count = UInt32(label == k)
    countf = Float32(count)
    (v * countf, count)
end
dotplus(ab0, ab1) = (ab1[1] + ab0[1], ab1[2] + ab0[2]) # could be .+ on 0.6

function shiftmeans!(data, means, labels)
    T = eltype(means)
    tmp = zeros(T, length(means))
    for k = 1:length(means)
        sum, len = mapreduce(
            meancount, dotplus,
            (zero(T), UInt32(0)), data, labels, UInt32(k)
        )
        tmp[k] = sum / (len + 1e-5)
    end
    copy!(means, tmp)
    nothing
end
function clustersmoved(prev_means, thrr_means)
    mapreduce(
        (a, b)-> UInt32(a != b), +,
        UInt32(0), prev_means, thrr_means
    )
end
function kmeans{T <: AbstractAccArray}(A::T, initialmeans, iter = 10)
    means = identity.(initialmeans)
    n = length(A); CT = eltype(T); ET = eltype(CT); k = length(means)
    labels = map(x-> UInt8(0), A)
    prevlabels = identity.(labels)
    for i = 1:iter
        copy!(prevlabels, labels)
        labels .= labeldata.(A, Base.RefValue(means), Int(k))
        num_changed = clustersmoved(prevlabels, labels)
        if num_changed < (n / 1000) + 1
            return means, labels
        end
        shiftmeans!(A, means, labels)
        @show i
    end
    means, labels
end

impath = homedir()*"/juliastuff/HurricaneKatrina/images/1.jpg"
clusters = [
    RGB{Float32}(0, 0, 0), RGB{Float32}(1, 0, 0),
    RGB{Float32}(0, 1, 0), RGB{Float32}(0, 0, 1),
    RGB{Float32}(1, 1, 0), RGB{Float32}(0, 1, 1),
]
imgs = RGB{Float32}.(load(impath)) # scale and convert image
imgvec = vec(imgs)

CUBackend.init()
thimgvec, thclusters = GPUArray(imgvec), GPUArray(clusters)

@time begin
    x,l = kmeans(thimgvec, thclusters)
    GPUArrays.synchronize(x)
end

tht1 = @elapsed kmeans(thimgvec, thclusters)
tht2 = @elapsed kmeans(thimgvec, thclusters)

using GPUArrays: buffer

f = (a, b)-> UInt32(a != b)
op = +
labels = map(x-> UInt8(0), thimgvec)
prevlabels = identity.(labels)

v0 = UInt32(0)
out = similar(buffer(labels), UInt32, (10,))
CUDAnative.@cuda (1, 1) CUBackend.reduce_kernel(out, f, op, v0, buffer(labels), buffer(prevlabels))
ast = CUDAnative.@code_typed CUDAnative.@cuda (1, 1) CUBackend.reduce_kernel(out, f, op, v0, buffer(labels), buffer(prevlabels))
code = ast[1][1].code
code_clean = filter(x-> !(Base.is_linenumber(x) || (isa(x, Expr) && (x.head in (:meta, :inbounds)))), code)
for elem in code_clean
    println(elem)
end
ci = ast[1][1]
for (T, name) in zip(ci.slottypes, ci.slotnames)
    if T == Any
        println(name)
    end
end
ast = Sugar.normalize_ast(code_clean)
body = Sugar.remove_goto(filter(x-> x != nothing && !Base.is_linenumber(x), ast))

body = Expr(:block)
append!(body.args, ast)
body

using CUDAnative, CUDAdrv
dev = CUDAdrv.CuDevice(0)
ctx = CUDAdrv.CuContext(dev)

len = 10^7
input = ones(Int32, len)

# CPU
cpu_val = reduce(+, input)


gpu_input = CuArray(input)
gpu_output = similar(gpu_input)

#
# Main implementation
#

# Reduce a value across a warp
@inline function reduce_warp(op::Function, val::T)::T where {T}
    offset = CUDAnative.warpsize() ÷ UInt32(2)
    # TODO: this can be unrolled if warpsize is known...
    while offset > 0
        val = op(val, shfl_down(val, offset))
        offset ÷= UInt32(2)
    end
    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op::Function, val::T)::T where {T}
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)

    # TODO: use fldmod1 (JuliaGPU/CUDAnative.jl#28)
    wid  = div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : zero(T)

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end

# Reduce an array across a complete grid
function reduce_grid(op::Function, input::CuDeviceVector{T}, output::CuDeviceVector{T},
                     len::Integer) where {T}

    # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
    i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= len
        @inbounds val = op(val, input[i])
        i += step
    end

    val = reduce_block(op, val)

    if threadIdx().x == UInt32(1)
        @inbounds output[blockIdx().x] = val
    end

    return
end

"""
Reduce a large array.
Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduce(op::Function, input::CuVector{T}, output::CuVector{T}) where {T}
    len = length(input)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 512
    blocks = min((len + threads - 1) ÷ threads, 1024)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    if length(output) < blocks
        throw(ArgumentError("output array too small, should be at least $blocks elements"))
    end

    CUDAnative.@code_typed @cuda (blocks,threads) reduce_grid(op, input, output, Int32(len))
end

gpu_reduce(+, gpu_input, gpu_output)



using CUDAnative, CUDAdrv, Colors
dev = CUDAdrv.CuDevice(0)
ctx = CUDAdrv.CuContext(dev)

@inline function reduce_block{T}(v0::T)
    shared = CUDAnative.@cuStaticSharedMem(T, 32)
    @inbounds shared[Cuint(1)] = v0
    return
end

@cuda (1, 1) reduce_block((0f0, 0f0, 0f0))
sizeof(RGB{Float32})
