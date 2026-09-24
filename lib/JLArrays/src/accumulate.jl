# scans

# As for sorting: Base, applied to the storage, with the method shapes of GPUArrays.

function Base._accumulate!(op, B::AnyJLArray, A::AnyJLVector, dims::Nothing, init::Nothing)
    accumulate!(op, _host(B), _host(A))
    return B
end
function Base._accumulate!(op, B::AnyJLArray, A::AnyJLVector, dims::Nothing, init::Some)
    accumulate!(op, _host(B), _host(A); init=something(init))
    return B
end
function Base._accumulate!(op, B::AnyJLArray, A::AnyJLArray, dims::Integer, init::Nothing)
    accumulate!(op, _host(B), _host(A); dims)
    return B
end
function Base._accumulate!(op, B::AnyJLArray, A::AnyJLArray, dims::Integer, init::Some)
    accumulate!(op, _host(B), _host(A); dims, init=something(init))
    return B
end
