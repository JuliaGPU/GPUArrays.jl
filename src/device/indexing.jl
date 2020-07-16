# indexing

export global_size, synchronize_threads, linear_index


# thread indexing functions
for f in (:blockidx, :blockdim, :threadidx, :griddim)
    @eval $f(ctx::AbstractKernelContext)::Int = error("Not implemented") # COV_EXCL_LINE
    @eval export $f
end

"""
    global_size(ctx::AbstractKernelContext)

Global size == blockdim * griddim == total number of kernel execution
"""
@inline function global_size(ctx::AbstractKernelContext)
    griddim(ctx) * blockdim(ctx)
end

"""
    linear_index(ctx::AbstractKernelContext)

linear index corresponding to each kernel launch (in OpenCL equal to get_global_id).

"""
@inline function linear_index(ctx::AbstractKernelContext)
    (blockidx(ctx) - 1) * blockdim(ctx) + threadidx(ctx)
end

"""
    linearidx(A, ctxsym = :ctx)

Macro form of `linear_index`, which calls return when out of bounds.
So it can be used like this:

    ```julia
    function kernel(ctx::AbstractKernelContext, A)
        idx = @linearidx A ctx
        # from here on it's save to index into A with idx
        @inbounds begin
            A[idx] = ...
        end
    end
    ```
"""
macro linearidx(A, ctxsym = :ctx)
    quote
        x1 = $(esc(A))
        i1 = linear_index($(esc(ctxsym)))
        i1 > length(x1) && return
        i1
    end
end

"""
    cartesianidx(A, ctxsym = :ctx)

Like [`@linearidx(A, ctxsym = :ctx)`](@ref), but returns a N-dimensional `CartesianIndex`.
"""
macro cartesianidx(A, ctxsym = :ctx)
    quote
        x = $(esc(A))
        i = @linearidx(x, $(esc(ctxsym)))
        @inbounds CartesianIndices(x)[i]
    end
end
