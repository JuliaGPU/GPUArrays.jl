# broadcasting operations

using Base.Broadcast

import Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

# but make sure we don't dispatch to the optimized copy method that directly indexes
function Broadcast.copy(bc::Broadcasted{<:AbstractGPUArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{}
        # using a Union{} eltype would fail early, during GPU array construction,
        # so use Nothing instead to give the error a chance to be thrown dynamically.
        ElType = Nothing
    end
    dest = copyto!(similar(bc, ElType), bc)
    return @allowscalar dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

# we need to override the outer copy method to make sure we never fall back to scalar
# iteration (see, e.g., CUDA.jl#145)
@inline function Broadcast.copy(bc::Broadcasted{<:AbstractGPUArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{}
        # using a Union{} eltype would fail early, during GPU array construction,
        # so use Nothing instead to give the error a chance to be thrown dynamically.
        ElType = Nothing
    end
    copyto!(similar(bc, ElType), bc)
end

@inline function Base.materialize!(::Style, dest, bc::Broadcasted) where {Style<:AbstractGPUArrayStyle}
    return _copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))))
end

@inline Base.copyto!(dest::AnyGPUArray, bc::Broadcasted{Nothing}) = _copyto!(dest, bc) # Keep it for ArrayConflict

@inline Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractGPUArrayStyle}) = _copyto!(dest, bc)

@inline function _copyto!(dest::AbstractArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    bc′ = Broadcast.preprocess(dest, bc)

    # grid-stride kernel
    function broadcast_kernel(ctx, dest, bc′, nelem)
        i = 0
        while i < nelem
            i += 1
            I = @cartesianidx(dest, i)
            @inbounds dest[I] = bc′[I]
        end
        return
    end
    elements = length(dest)
    elements_per_thread = typemax(Int)
    heuristic = launch_heuristic(backend(dest), broadcast_kernel, dest, bc′, 1;
                                 elements, elements_per_thread)
    config = launch_configuration(backend(dest), heuristic;
                                  elements, elements_per_thread)
    gpu_call(broadcast_kernel, dest, bc′, config.elements_per_thread;
             threads=config.threads, blocks=config.blocks)

    return dest
end

## map

allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)

function Base.map(f, x::AnyGPUArray, xs::AbstractArray...)
    # if argument sizes match, their shape needs to be preserved
    xs = (x, xs...)
    if allequal(size.(xs)...)
         return f.(xs...)
    end

    # if not, treat them as iterators
    indices = LinearIndices.(xs)
    common_length = minimum(length.(indices))

    # construct a broadcast to figure out the destination container
    ElType = Broadcast.combine_eltypes(f, xs)
    isbitstype(ElType) || error("Cannot map function returning non-isbits $ElType.")
    dest = similar(x, ElType, common_length)

    return map!(f, dest, xs...)
end

function Base.map!(f, dest::AnyGPUArray, xs::AbstractArray...)
    # custom broadcast, ignoring the container size mismatches
    # (avoids the reshape + view that our mapreduce impl has to do)
    indices = LinearIndices.((dest, xs...))
    common_length = minimum(length.(indices))
    common_length==0 && return

    bc = Broadcast.instantiate(Broadcast.broadcasted(f, xs...))
    if bc isa Broadcast.Broadcasted
        bc = Broadcast.preprocess(dest, bc)
    end

    # grid-stride kernel
    function map_kernel(ctx, dest, bc, nelem)
        for i in 1:nelem
            j = linear_index(ctx, i)
            j > common_length && return

            J = CartesianIndices(axes(bc))[j]
            @inbounds dest[j] = bc[J]
        end
        return
    end
    elements = common_length
    elements_per_thread = typemax(Int)
    heuristic = launch_heuristic(backend(dest), map_kernel, dest, bc, 1;
                                 elements, elements_per_thread)
    config = launch_configuration(backend(dest), heuristic;
                                  elements, elements_per_thread)
    gpu_call(map_kernel, dest, bc, config.elements_per_thread;
             threads=config.threads, blocks=config.blocks)

    return dest
end
