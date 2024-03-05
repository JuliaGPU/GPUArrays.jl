# indexing

export global_index, global_size, linear_index, @linearidx, @cartesianidx


## hardware

for f in (:blockidx, :blockdim, :threadidx, :griddim)
    @eval $f(ctx::AbstractKernelContext)::Int = error("Not implemented") # COV_EXCL_LINE
    @eval export $f
end

"""
    global_index(ctx::AbstractKernelContext)

Query the global index of the current thread in the launch configuration (i.e. as far as the
hardware is concerned).
"""
@inline function global_index(ctx::AbstractKernelContext)
    threadidx(ctx) + (blockidx(ctx) - 1) * blockdim(ctx)
end

"""
    global_size(ctx::AbstractKernelContext)

Query the global size of the launch configuration (total number of threads launched).
"""
@inline function global_size(ctx::AbstractKernelContext)
    griddim(ctx) * blockdim(ctx)
end


## logical

"""
    linear_index(ctx::AbstractKernelContext, grididx::Int=1)

Return a linear index for the current kernel by querying hardware registers (similar to
`get_global_id` in OpenCL). For applying a grid stride (in terms of [`global_size`](@ref)),
specify `grididx`.

"""
@inline function linear_index(ctx::AbstractKernelContext, grididx::Int=1)
    global_index(ctx) + (grididx - 1) * global_size(ctx)
end

"""
    linearidx(A, grididx=1, ctxsym=:ctx)

Macro form of [`linear_index`](@ref), which return from the surrouunding scope when out of
bounds:

    ```julia
    function kernel(ctx::AbstractKernelContext, A)
        idx = @linearidx A
        # from here on it's safe to index into A with idx
        @inbounds begin
            A[idx] = ...
        end
    end
    ```
"""
macro linearidx(A, grididx=1, ctxsym=:ctx)
    quote
        x = $(esc(A))
        i = linear_index($(esc(ctxsym)), $(esc(grididx)))
        if !(1 <= i <= length(x))
            return
        end
        i
    end
end

"""
    cartesianidx(A, grididx=1, ctxsym=:ctx)

Like [`@linearidx`](@ref), but returns a N-dimensional `CartesianIndex`.
"""
macro cartesianidx(A, grididx=1, ctxsym=:ctx)
    quote
        x = $(esc(A))
        i = @linearidx(x, $(esc(grididx)), $(esc(ctxsym)))
        @inbounds CartesianIndices(x)[i]
    end
end


## utilities

# indexing a CartesianIndices at run time generates an integer division, which is slow.
# to work around this, we can use a static CartesianIndices type to avoid the division.
# this informs LLVM and the back-end about the static iteration bounds, allowing it to
# lower the integer divison to a series of bit shifts, dramatically improving performance.
# also see: maleadt/StaticCartesian.jl, JuliaGPU/GPUArrays.jl#454

struct StaticCartesianIndices{N, I} end

StaticCartesianIndices(iter::CartesianIndices{N}) where {N} =
    StaticCartesianIndices{N, iter.indices}()
StaticCartesianIndices(x) = StaticCartesianIndices(CartesianIndices(x))

Base.CartesianIndices(iter::StaticCartesianIndices{N, I}) where {N, I} =
    CartesianIndices{N, typeof(I)}(I)

Base.@propagate_inbounds Base.getindex(I::StaticCartesianIndices, i::Int) =
    CartesianIndices(I)[i]
Base.length(I::StaticCartesianIndices) = length(CartesianIndices(I))

function Base.show(io::IO, I::StaticCartesianIndices)
    print(io, "Static")
    show(io, CartesianIndices(I))
end
