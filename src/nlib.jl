using GPUArrays, MacroTools

# unroll helper
"""
Unrolls like ntuple, but allows to pass args -
this helps with avoiding boxing due to closure capturing
"""
@generated function arg_ntuple(f::F, n::Type{Val{N}}, args::Vararg{Any, N2}) where {F, N, N2}
    args_un = ntuple(i-> :(args[$i]), N2)
    tup = ntuple(i-> :(f($i, $(args_un...))), N)
    quote
        Base.@_inline_meta
        ($(tup...),)
    end
end


"""
Like arg_ntuple, but offers a reduction variable v0, that doesn't get boxed as it
would be the case with a closure.
"""
@generated function nreduce(f, ::Type{Val{N}}, v0, args::NTuple{N2, Any}) where {N, N2}
    args_un = ntuple(i-> :(args[$i]), N2)
    reduced = ntuple(i-> :(v0 = f($i, v0, $(args_un...))), N)
    quote
        Base.@_inline_meta
        $(reduced...)
    end
end

"""
Unroll macro for static ranges
"""
macro unroll(forloop)
    @capture(forloop, for idx_ in from_ : to_; body__; end) || throw(
        ArgumentError("Needs to be a for loop. Found: $forloop")
    )
    expr = Expr(:block); body = esc.(body)
    expr.args = map(from:to) do i
        :($(esc(idx)) = $i; $(body...))
    end
end


function blockReduce(
        state,
        smem::AbstractArray{AccumT},
        val::AccumT,
        op,
        v0::AccumT
    ) where AccumT
    # To avoid RaW races from chaining blockReduce calls together, we need a sync here
    synchronize_threads(state)

    smem[threadidx_x(state)] = val

    synchronize_threads(state)

    warpVal = v0;
    # First warp will perform per-warp reductions for the remaining warps
    ui32 = UInt32(32)
    ui1 = UInt32(1)
    if threadidx_x(state) < ui32
        lane = (threadidx_x(state) - ui1) % ui32
        if lane < blockdim_x(state) ÷ ui32
            @unroll for idx in 1:32
                warpVal = op(warpVal, smem[lane * ui32 + UInt32(idx)])
            end
            smem[ui1 + lane] = warpVal
        end
    end

    synchronize_threads(state)

    # First thread will perform a reduction of the above per-warp reductions
    blockVal = v0

    if threadidx_x(state) == ui1
        for i in ui1:(((blockdim_x(state) - ui1) ÷ ui32) + ui1)
            blockVal = op(blockVal, smem[i])
        end
        smem[ui1] = blockVal
    end

    # Sync and broadcast
    synchronize_threads(state)
    return smem[ui1]
end

function ilpReduce(
        state,
        data::AbstractArray{T},
        data_offset,
        size,
        op,
        v0::AccumT,
        ilp::Type{Val{ILP}}
    ) where {T, AccumT, ILP}

    ILP32 = UInt32(ILP)
    threadVal = v0
    ui1 = UInt32(1)
    offset = threadidx_x(state) - ui1

    last = size % (ILP32 * blockdim_x(state))

    # Body (unroll by ILP times)
    while offset < (size - last)
        tmp = arg_ntuple(ilp, ui1 + data_offset + offset, data) do j, off, d
            d[ui1 + off + UInt32(j - 1) * blockdim_x(state)]
        end
        threadVal = nreduce(ilp, threadVal, tmp) do idx, v0, a
            op(v0, a[idx])
        end
        offset += blockdim_x(state) * ILP32
    end

    # Epilogue
    while offset < size
        threadVal = op(threadVal, data[ui1 + data_offset + offset])
        offset += blockdim_x(state)
    end

    return threadVal
end



function softmax_forward(
        state,
        output::AbstractArray{T},
        input::AbstractArray{T},
        ::Type{AccumT},
        classes::UInt32,
        ::Type{Val{SMemSize}},
        ilp::Type{Val{ILP}},
        ::Type{Epilogue}
    ) where {T, AccumT, SMemSize, ILP, Epilogue}

    ui1 = UInt32(1)
    ILP32 = UInt32(ILP)
    smem = @LocalMemory(state, AccumT, SMemSize)
    # forward pointers to batch[blockidx_x(state)]
    # each block handles a sample in the mini-batch
    global_offset = (blockidx_x(state) - ui1) * classes;

    # find the max
    thread_max = ilpReduce(state, input, global_offset, classes, max, -typemax(AccumT), ilp)
    max_k = blockReduce(state, smem, thread_max, max, -typemax(AccumT))
    max_k_non_accum = T(max_k)

    # reduce all values
    threadExp = ilpReduce(
        state, input, global_offset, classes,
        (sum, v)-> sum + exp(v - max_k_non_accum), AccumT(0), ilp
    )
    sumAll = blockReduce(state, smem, threadExp, +, AccumT(0))

    offset = threadidx_x(state) - ui1
    last = classes % (ILP32 * blockdim_x(state))
    epilogue = Epilogue(max_k_non_accum, sumAll)
    while offset < (classes - last)
        tmp = arg_ntuple(ilp, ui1 + global_offset + offset) do j, off
            input[ui1 + off + UInt32(j - 1) * blockdim_x(state)]
        end
        for j in ui1:ILP32
            idx = global_offset + offset + UInt32(j - 1) * blockdim_x(state)
            output[ui1 + idx] = apply(epilogue, tmp[j])
        end
        offset += blockdim_x(state) * ILP32
    end

    while offset < classes
        output[ui1 + offset] = apply(epilogue, input[ui1 + global_offset + offset])
        offset += blockdim_x(state)
    end
end


function getblocksize(ILP::Integer, dim_size::Integer)
    block_size = 1;
    max_block_size = min(dim_size ÷ ILP, 1024)
    while block_size < max_block_size; block_size *= 2; end
    # Launch at least a single warp - the kernel assumes that.
    block_size = max(block_size, 32)
    return (block_size,)
end

struct SoftMaxForwardEpilogue{T1, T2}
    sum::T1
    max_input::T2
end
apply(x::SoftMaxForwardEpilogue, input) = exp(input - x.max_input) / x.sum

function HostSoftMaxForward(
        input::AbstractArray{T}, output::AbstractArray{T},
        outer_size::Int, dim_size::Int, inner_size::Int, dim::Int,
        Epilogue = SoftMaxForwardEpilogue
    ) where T
    # This kernel spawns a block per each element in the batch.
    # XXX: it assumes that inner_size == 1
    if inner_size == 1
        AccumT = T # TODO calculate output type
        ILP = 2
        grid = (outer_size,)
        block = getblocksize(ILP, dim_size)
        SMemSize = block[1]
        gpu_call(
            softmax_forward, input,
            (output, input, AccumT, UInt32(dim_size), Val{SMemSize}, Val{ILP}, Epilogue),
            (grid, block)
        )
        # This kernel runs in a 2D grid, where each application along y dimension has a fixed
        # outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
        # Reductions over dim are done in a single-threaded manner.
    else
        # uint32_t smem_size;
        # dim3 grid, block;
        # SpatialSoftMax_getLaunchSizes(
        #     state, &cunn_SpatialSoftMaxForward<T, AccumT, Epilogue>,
        #     outer_size, dim_size, inner_size,
        #     grid, block, smem_size
        # );
        #
        # cunn_SpatialSoftMaxForward<T, AccumT, Epilogue>
        # <<<grid, block, smem_size, THCState_getCurrentStream(state)>>>(
        #     output, input, outer_size, dim_size, inner_size
        # )
    end
end
using CLArrays
input = CLArray(rand(2, 3, 4, 5, 6))
output = CLArray(rand(100))

HostSoftMaxForward(
    input, output,
    30, 4, 6, 3,
    SoftMaxForwardEpilogue
)

input2 = JLArray(Array(input))
output2 = JLArray(zeros(100))
HostSoftMaxForward(
    input2, output2,
    30, 4, 6, 3,
    SoftMaxForwardEpilogue
)
Array(output2)
Array(output)[1:2]

# m = Transpiler.CLMethod((softmax_forward,
#     (CLArrays.KernelState, CLArrays.DeviceArray{Float64,2,Transpiler.CLIntrinsics.GlobalPointer{Float64}},
#     CLArrays.DeviceArray{Float64,2,Transpiler.CLIntrinsics.GlobalPointer{Float64}},
#     Type{Float64}, UInt32, Type{Val{2}}, Type{Val{2}}, Type{SoftMaxForwardEpilogue})
# ))
# Sugar.getast!(m)
#
# expr = Sugar.sugared(m.signature..., code_typed)
# println(expr)
# expr = Sugar.code_typed(m.signature..., optimize = false)[1][1].code
# typs = Sugar.expr_type.(m, expr[45].args[2].args[2:end])
# typs
# expr = Sugar.code_typed(arg_ntuple, (typs...), optimize = true)
# Sugar.getast!(Transpiler.CLMethod(typs[1], (typs[2:end]...)))
# arg_ntuple(Val{UInt})
# Sugar.show_source(STDERR, m, expr)
