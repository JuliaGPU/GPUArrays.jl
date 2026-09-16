# sorting

# JLArrays are backed by a regular Array, so the sorting primitives defer to Base on that storage.

function GPUArrays.sort!(A::AnyJLArray; dims, lt, by, rev, order)
    a = typed_data(A)
    @allowscalar if dims === Colon()
        sort!(vec(a); lt, by, rev, order)
    else
        sort!(a; dims, lt, by, rev, order)
    end
    A
end

function GPUArrays.sortperm!(ix::AnyJLArray, A::AnyJLArray; dims, lt, by, rev, order, initialized)
    @allowscalar if dims === Colon()
        sortperm!(vec(typed_data(ix)), vec(typed_data(A)); lt, by, rev, order, initialized)
    else
        sortperm!(typed_data(ix), typed_data(A); dims, lt, by, rev, order, initialized)
    end
    ix
end
