indexlength(A, i, array::AbstractArray) = length(array)
indexlength(A, i, array::Number) = 1
indexlength(A, i, array::Colon) = size(A, i)

function Base.setindex!{T, N}(A::GPUArray{T, N}, value, indexes...)
    # similarly, value should always be a julia array
    shape = ntuple(Val{N}) do i
        indexlength(A, i, indexes[i])
    end
    if !isa(value, T) # TODO, shape check errors for x[1:3] = 1
        Base.setindex_shape_check(value, indexes...)
    end
    checkbounds(A, indexes...)
    v = array_convert(Array{T, N}, value)
    # since you shouldn't update GPUArrays with single indices, we simplify the interface
    # by always mapping to ranges
    ranges_dest = to_cartesian(A, indexes)
    ranges_src = CartesianRange(size(v))

    copy!(A, ranges_dest, v, ranges_src)
    return
end

Base.getindex{T}(A::GPUArray{T, 0}) = Array(A)[]

function Base.getindex{T, N}(A::GPUArray{T, N}, indexes...)
    cindexes = Base.to_indices(A, indexes)
    # similarly, value should always be a julia array
    # We shouldn't really bother about checkbounds performance, since setindex/getindex will always be relatively slow
    checkbounds(A, cindexes...)

    shape = map(length, cindexes)
    result = Array{T, length(shape)}(shape)
    ranges_src = to_cartesian(A, cindexes)
    ranges_dest = CartesianRange(shape)
    copy!(result, ranges_dest, A, ranges_src)
    if all(i-> isa(i, Integer), cindexes) # scalar
        return result[]
    else
        return result
    end
end
