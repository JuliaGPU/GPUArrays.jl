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


# Base functions that are sadly not fit for the the GPU yet (they only work for Int64)
Base.@pure @inline function gpu_ind2sub(A::AbstractArray, ind::T) where T
    _ind2sub(size(A), ind - T(1))
end
Base.@pure @inline function gpu_ind2sub(dims::NTuple{N}, ind::T) where {N, T}
    _ind2sub(NTuple{N, T}(dims), ind - T(1))
end
Base.@pure @inline _ind2sub(::Tuple{}, ind::T) where {T} = (ind + T(1),)
Base.@pure @inline function _ind2sub(indslast::NTuple{1}, ind::T) where T
    ((ind + T(1)),)
end
Base.@pure @inline function _ind2sub(inds, ind::T) where T
    r1 = inds[1]
    indnext = div(ind, r1)
    f = T(1); l = r1
    (ind-l*indnext+f, _ind2sub(Base.tail(inds), indnext)...)
end

Base.@pure function gpu_sub2ind(dims::NTuple{N}, I::NTuple{N2, T}) where {N, N2, T}
    Base.@_inline_meta
    _sub2ind(NTuple{N, T}(dims), T(1), T(1), I...)
end
_sub2ind(x, L, ind) = ind
function _sub2ind(::Tuple{}, L, ind, i::T, I::T...) where T
    Base.@_inline_meta
    ind + (i - T(1)) * L
end
function _sub2ind(inds, L, ind, i::IT, I::IT...) where IT
    Base.@_inline_meta
    r1 = inds[1]
    _sub2ind(Base.tail(inds), L * r1, ind + (i - IT(1)) * L, I...)
end

"""
    cartesianidx(A, ctxsym = :ctx)

Like [`@linearidx(A, ctxsym = :ctx)`](@ref), but returns a N-dimensional `CartesianIndex`.
"""
macro cartesianidx(A, ctxsym = :ctx)
    quote
        x = $(esc(A))
        i = @linearidx(x, $(esc(ctxsym)))
        CartesianIndices(x)[i]
    end
end
