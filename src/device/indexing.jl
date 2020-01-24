# indexing

export global_size, synchronize_threads, linear_index


# thread indexing functions
for sym in (:x, :y, :z)
    for f in (:blockidx, :blockdim, :threadidx, :griddim)
        fname = Symbol(string(f, '_', sym))
        @eval $fname(state)::Int = error("Not implemented") # COV_EXCL_LINE
        @eval export $fname
    end
end

"""
    global_size(state)

Global size == blockdim * griddim == total number of kernel execution
"""
@inline function global_size(state)
    # TODO nd version
    griddim_x(state) * blockdim_x(state)
end

"""
    linear_index(state)

linear index corresponding to each kernel launch (in OpenCL equal to get_global_id).

"""
@inline function linear_index(state)
    (blockidx_x(state) - 1) * blockdim_x(state) + threadidx_x(state)
end

"""
    linearidx(A, statesym = :state)

Macro form of `linear_index`, which calls return when out of bounds.
So it can be used like this:

    ```julia
    function kernel(state, A)
        idx = @linear_index A state
        # from here on it's save to index into A with idx
        @inbounds begin
            A[idx] = ...
        end
    end
    ```
"""
macro linearidx(A, statesym = :state)
    quote
        x1 = $(esc(A))
        i1 = linear_index($(esc(statesym)))
        i1 > length(x1) && return
        i1
    end
end

"""
    cartesianidx(A, statesym = :state)

Like [`@linearidx(A, statesym = :state)`](@ref), but returns an N-dimensional `NTuple{ndim(A), Int}` as index
"""
macro cartesianidx(A, statesym = :state)
    quote
        x = $(esc(A))
        i2 = @linearidx(x, $(esc(statesym)))
        gpu_ind2sub(x, i2)
    end
end
