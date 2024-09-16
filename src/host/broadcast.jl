# broadcasting operations

using Base.Broadcast

using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

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
    if ElType == Union{} || !Base.allocatedinline(ElType)
        # a Union{} or non-isbits eltype would fail early, during GPU array construction,
        # so use a special marker to give the error a chance to be thrown during compilation
        # or even dynamically, and pick that marker up afterwards to throw an error.
        ElType = BrokenBroadcast{ElType}
    end
    copyto!(similar(bc, ElType), bc)
end

struct BrokenBroadcast{T} end
Base.convert(::Type{BrokenBroadcast{T}}, x) where T = BrokenBroadcast{T}()
Base.convert(::Type{BrokenBroadcast{T}}, x::BrokenBroadcast{T}) where T = x
Base.eltype(::Type{BrokenBroadcast{T}}) where T = T

@inline function Base.materialize!(::Style, dest, bc::Broadcasted) where {Style<:AbstractGPUArrayStyle}
    return _copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))))
end

@inline Base.copyto!(dest::AnyGPUArray, bc::Broadcasted{Nothing}) =
    _copyto!(dest, bc) # Keep it for ArrayConflict

@inline Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractGPUArrayStyle}) =
    _copyto!(dest, bc)

@inline function _copyto!(dest::AbstractArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    bc = Broadcast.preprocess(dest, bc)

    @kernel function broadcast_kernel_linear(dest, bc)
        I = @index(Global, Linear)
        @inbounds dest[I] = bc[I]
    end

    @kernel function broadcast_kernel_cartesian(dest, bc)
        I = @index(Global, Cartesian)
        @inbounds dest[I] = bc[I]
    end

    # grid-stride kernel, ndrange set for possible 0D evaluation
    if ndims(dest) == 1 || (isa(IndexStyle(dest), IndexLinear) &&
                           isa(IndexStyle(bc), IndexLinear))
        broadcast_kernel_linear(get_backend(dest))(dest, bc;
            ndrange = length(size(dest)) > 0 ? length(dest) : 1)
    else
        broadcast_kernel_cartesian(get_backend(dest))(dest, bc;
            ndrange = sz = length(size(dest)) > 0 ? size(dest) : (1,))
    end

    if eltype(dest) <: BrokenBroadcast
        throw(ArgumentError("Broadcast operation resulting in $(eltype(eltype(dest))) is not GPU compatible"))
    end

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
    if ElType == Union{} || !Base.allocatedinline(ElType)
        # see `broadcast`
        ElType = BrokenBroadcast{ElType}
    end
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
    @kernel function map_kernel(dest, bc, nelem, common_length)

        j = 0
        J = @index(Global, Linear)
        for i in 1:nelem
            j += 1
            if j <= common_length

                J_c = CartesianIndices(axes(bc))[(J-1)*nelem + j]
                @inbounds dest[J_c] = bc[J_c]
            end
        end
    end
    elements = common_length
    elements_per_thread = typemax(Int)
    kernel = map_kernel(get_backend(dest))
    heuristic = launch_heuristic(get_backend(dest), kernel, dest, bc, 1,
                                 common_length; elements, elements_per_thread)
    config = launch_configuration(get_backend(dest), heuristic;
                                  elements, elements_per_thread)
    kernel(dest, bc, config.elements_per_thread,
           common_length; ndrange = config.threads)

    if eltype(dest) <: BrokenBroadcast
        throw(ArgumentError("Map operation resulting in $(eltype(eltype(dest))) is not GPU compatible"))
    end

    return dest
end
