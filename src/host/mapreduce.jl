# map-reduce

const AbstractArrayOrBroadcasted = Union{AbstractArray,Broadcast.Broadcasted}

# # GPUArrays' mapreduce methods build on `Base.mapreducedim!`, but with an additional
# # argument `init` value to avoid eager initialization of `R` (if set to something).
# mapreducedim!(f, op, R::AnyGPUArray, A::AbstractArrayOrBroadcasted;
#               init=nothing) = error("Not implemented") # COV_EXCL_LINE
# resolve ambiguities
Base.mapreducedim!(f, op, R::AnyGPUArray, A::AbstractArray) = mapreducedim!(f, op, R, A)
Base.mapreducedim!(f, op, R::AnyGPUArray, A::Broadcast.Broadcasted) = mapreducedim!(f, op, R, A)

neutral_element(op, T) =
    error("""GPUArrays.jl needs to know the neutral element for your operator `$op`.
             Please pass it as an explicit argument to `GPUArrays.mapreducedim!`,
             or register it globally by defining `GPUArrays.neutral_element(::typeof($op), T)`.""")
neutral_element(::typeof(Base.:(|)), T) = zero(T)
neutral_element(::typeof(Base.:(+)), T) = zero(T)
neutral_element(::typeof(Base.add_sum), T) = zero(T)
neutral_element(::typeof(Base.:(&)), T) = one(T)
neutral_element(::typeof(Base.:(*)), T) = one(T)
neutral_element(::typeof(Base.mul_prod), T) = one(T)
neutral_element(::typeof(Base.min), T) = typemax(T)
neutral_element(::typeof(Base.max), T) = typemin(T)
neutral_element(::typeof(Base._extrema_rf), ::Type{<:NTuple{2,T}}) where {T} = typemax(T), typemin(T)

# resolve ambiguities
Base.mapreduce(f, op, A::AnyGPUArray, As::AbstractArrayOrBroadcasted...;
               dims=:, init=nothing) = _mapreduce(f, op, A, As...; dims=dims, init=init)
Base.mapreduce(f, op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, As::AbstractArrayOrBroadcasted...;
               dims=:, init=nothing) = _mapreduce(f, op, A, As...; dims=dims, init=init)

function _mapreduce(f::F, op::OP, As::Vararg{Any,N}; dims::D, init) where {F,OP,N,D}
    # figure out the destination container type by looking at the initializer element,
    # or by relying on inference to reason through the map and reduce functions
    if init === nothing
        ET = Broadcast.combine_eltypes(f, As)
        ET = Base.promote_op(op, ET, ET)
        (ET === Union{} || ET === Any) &&
            error("mapreduce cannot figure the output element type, please pass an explicit init value")

        init = neutral_element(op, ET)
    else
        ET = typeof(init)
    end

    # apply the mapping function to the input arrays
    if N == 1
        # ... with only a single input, we can defer this to the reduce step
        A = only(As)
    else
        # mapreduce should apply `f` like `map` does, consuming elements like iterators
        A = if allequal(size.(As)...)
            Broadcast.instantiate(Broadcast.broadcasted(f, As...))
        else
            # TODO: can we avoid the reshape + view?
            indices = LinearIndices.(As)
            common_length = minimum(length.(indices))
            Bs = map(As) do A
                view(reshape(A, length(A)), 1:common_length)
            end
            Broadcast.instantiate(Broadcast.broadcasted(f, Bs...))
        end
        f = identity
    end

    # allocate an output container
    sz = size(A)
    red = ntuple(i->(dims==Colon() || i in dims) ? 1 : sz[i], length(sz))
    R = similar(A, ET, red)

    # perform the reduction
    if prod(sz) == 0
        fill!(R, init)
    else
        mapreducedim!(f, op, R, A; init)
    end

    # return the result
    if dims === Colon()
        @allowscalar R[]
    else
        R
    end
end

Base.any(A::AnyGPUArray{Bool}) = mapreduce(identity, |, A)
Base.all(A::AnyGPUArray{Bool}) = mapreduce(identity, &, A)

Base.any(f::Function, A::AnyGPUArray) = mapreduce(f, |, A)
Base.all(f::Function, A::AnyGPUArray) = mapreduce(f, &, A)

Base.count(pred::Function, A::AnyGPUArray; dims=:, init=0) =
    mapreduce(pred, Base.add_sum, A; init=init, dims=dims)

# avoid calling into `initarray!`
for (fname, op) in [(:sum, :(Base.add_sum)), (:prod, :(Base.mul_prod)),
                    (:maximum, :(Base.max)), (:minimum, :(Base.min)),
                    (:all, :&),              (:any, :|)]
    fname! = Symbol(fname, '!')
    @eval begin
        Base.$(fname!)(f::Function, r::AnyGPUArray, A::AnyGPUArray{T}) where T =
            GPUArrays.mapreducedim!(f, $(op), r, A; init=neutral_element($(op), T))
    end
end

LinearAlgebra.ishermitian(A::AbstractGPUMatrix) = mapreduce(==, &, A, adjoint(A))


# comparisons

# ignores missing
function Base.isequal(A::AnyGPUArray, B::AnyGPUArray)
    if A === B return true end
    if axes(A) != axes(B)
        return false
    end
    mapreduce(isequal, &, A, B; init=true)
end

# returns `missing` when missing values are involved
function Base.:(==)(A::AnyGPUArray, B::AnyGPUArray)
    if axes(A) != axes(B)
        return false
    end

    function mapper(a, b)
        eq = (a == b)
        if ismissing(eq)
            (; is_missing=true, is_equal=#=don't care=#false)
        else
            (; is_missing=false, is_equal=eq)
        end
    end
    function reducer(a, b)
        if a.is_missing || b.is_missing
            (; is_missing=true, is_equal=#=don't care=#false)
        else
            (; is_missing=false, is_equal=a.is_equal & b.is_equal)
        end
    end
    res = mapreduce(mapper, reducer, A, B; init=(; is_missing=false, is_equal=true))
    res.is_missing ? missing : res.is_equal
end


import KernelAbstractions: @context

@inline function reduce_group(@context, op, val::T, neutral, ::Val{maxitems}) where {T, maxitems}
    items = @groupsize()[1]
    item = @index(Local, Linear)

    # local mem for a complete reduction
    shared = @localmem T (maxitems,)
    @inbounds shared[item] = val

    # perform a reduction
    d = 1
    while d < items
        @synchronize() # legal since cpu=false
        index = 2 * d * (item-1) + 1
        @inbounds if index <= items
            other_val = if index + d <= items
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first item
    if item == 1
        val = @inbounds shared[item]
    end

    return val
end

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
@kernel cpu=false function partial_mapreduce_device(f, op, neutral, maxitems, Rreduce, Rother, R, As...)
    # decompose the 1D hardware indices into separate ones for reduction (across items
    # and possibly groups if it doesn't fit) and other elements (remaining groups)
    localIdx_reduce = @index(Local, Linear)
    localDim_reduce = @groupsize()[1]
    groupIdx_reduce, groupIdx_other = fldmod1(@index(Group, Linear), length(Rother))
    numGroups = length(KernelAbstractions.blocks(KernelAbstractions.__iterspace(@context())))
    groupDim_reduce = numGroups ÷ length(Rother)

    # group-based indexing into the values outside of the reduction dimension
    # (that means we can safely synchronize items within this group)
    iother = groupIdx_other
    @inbounds if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = CartesianIndex(Tuple(Iother)..., groupIdx_reduce)
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end

        val = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a group
        ireduce = localIdx_reduce + (groupIdx_reduce - 1) * localDim_reduce
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = max(Iother, Ireduce)
            val = op(val, f(_map_getindex(As, J)...))
            ireduce += localDim_reduce * groupDim_reduce
        end

        val = reduce_group(@context(), op, val, neutral, maxitems)

        # write back to memory
        if localIdx_reduce == 1
            R[Iout] = val
        end
    end
end

## COV_EXCL_STOP

function mapreducedim!(f::F, op::OP, R::AnyGPUArray{T}, A::AbstractArrayOrBroadcasted;
                                 init=nothing) where {F, OP, T}
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)

    # allocate an additional, empty dimension to write the reduced value to.
    # this does not affect the actual location in memory of the final values,
    # but allows us to write a generalized kernel supporting partial reductions.
    R′ = reshape(R, (size(R)..., 1))

    # how many items do we want?
    #
    # items in a group work together to reduce values across the reduction dimensions;
    # we want as many as possible to improve algorithm efficiency and execution occupancy.
    wanted_items = length(Rreduce)
    function compute_items(max_items)
        if wanted_items > max_items
            max_items
        else
            wanted_items
        end
    end

    # how many items can we launch?
    #
    # we might not be able to launch all those items to reduce each slice in one go.
    # that's why each items also loops across their inputs, processing multiple values
    # so that we can span the entire reduction dimension using a single item group.

    # group size is restricted by local memory
    # max_lmem_elements = compute_properties(device()).maxSharedLocalMemory ÷ sizeof(T)
    # max_items = min(compute_properties(device()).maxTotalGroupSize,
    #                 compute_items(max_lmem_elements ÷ 2))
    # TODO: dynamic local memory to avoid two compilations

    # let the driver suggest a group size
    # args = (f, op, init, Val(max_items), Rreduce, Rother, R′, A)
    # kernel_args = kernel_convert.(args)
    # kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    # kernel = zefunction(partial_mapreduce_device, kernel_tt)
    # reduce_items = launch_configuration(kernel)
    reduce_items = 512
    reduce_kernel = partial_mapreduce_device(get_backend(R), (reduce_items,))

    # how many groups should we launch?
    #
    # even though we can always reduce each slice in a single item group, that may not be
    # optimal as it might not saturate the GPU. we already launch some groups to process
    # independent dimensions in parallel; pad that number to ensure full occupancy.
    other_groups = length(Rother)
    reduce_groups = cld(length(Rreduce), reduce_items)

    # determine the launch configuration
    items = reduce_items
    groups = reduce_groups*other_groups

    ndrange = groups*items

    # perform the actual reduction
    if reduce_groups == 1
        # we can cover the dimensions to reduce using a single group
        reduce_kernel(f, op, init, Val(items), Rreduce, Rother, R′, A; ndrange)
    else
        # we need multiple steps to cover all values to reduce
        partial = similar(R, (size(R)..., reduce_groups))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            partial .= R
        end
        reduce_kernel(f, op, init, Val(items), Rreduce, Rother, partial, A; ndrange)

        GPUArrays.mapreducedim!(identity, op, R′, partial; init=init)
    end

    return R
end
