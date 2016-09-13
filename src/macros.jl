
using MacroTools

macro gpu(forloop)
    i, range, body = @match forloop begin
        for i_ = range_
            body__
        end => (i, range, body)
        for i_ in range_
            body__
        end => (i, range, body)
    end
    argsym = gensym(:arg)
    iexpr = if isa(i, Expr) && i.head == :tuple
        Expr(:(=), :(($(i.args...),)), argsym)
    elseif isa(i, Symbol)
        :($i = $argsym)
    else
        error("$i not supported")
    end

    expr = quote
        foreach($range) do $argsym
            $iexpr
            $(Expr(:block, body...))
        end
    end
    dump(body)
    expr
end

A = rand(10)
B = rand(11)

@gpu(for (i,b) = enumerate(1:10)
    A[i] = i
    B[i+1] = b
end, A, B)

using CUDAnative, CUDAdrv
dev = CuDevice(0)
ctx = CuContext(dev)


@target ptx function gpu_foreach_kernel3(f, range)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds idx = range[i]
    f(idx)
    nothing
end
@target ptx function test2(cu2, cu1)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds cu2[i] = cu1[i] * i
    nothing
end
len = 50
threads = min(len, 1024)
blocks = floor(Int, len/threads)
@cuda (blocks, threads) test2(cu2, cu1)

function gpu_foreach(f, range)
    len = length(range)
    threads = min(len, 1024)
    blocks = floor(Int, len/threads)
    i32range = map(Int32, range)
    @code_warntype gpu_foreach_kernel3(f, i32range)
    @cuda (blocks, threads) gpu_foreach_kernel3(f, i32range)
end

const cu1 = CuArray(rand(50))
const cu2 = CuArray(Float64, 50)

gpu_foreach(1:50) do i
    @inbounds cu2[i] = cu1[i] * i
    nothing
end
@code_warntype(f(1:50))
# quick poll:
# I want to make working with GPU arrays easier so I want to write a macro which
# turns normal for loops into an equivalent parallel loop on the gpu.
# The problem is that you might want to work with indexes only in the loop header,
# which means the macro would go through extra lengths to figure out if gpu arrays
# are involved or not. So there are two options:
# Magical option: Go through all asignements and then do something like that:
#
# ```
# @gpu for i=1:10
#     ...
# end
# turns into:
# if is_gpu_array($(all_assignements...))
#     gpu_for(...)
# else
#     foreach(...)
# end
# ```
# Or I force the user to put the GPU array into the header:
# ```
# @gpu A for ...
# end
# ```
