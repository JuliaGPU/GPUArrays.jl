# printing, through a copy on the host

# the name of the array type that stores the entries: "MtlArray", "CuArray", ...
storage_name(A) = nameof(typeof(A.nzVal))

# the concrete type is long and not what users write (they use the back-end aliases), so
# only show the format and the element and index types
function Base.summary(io::IO, A::GPUSparseArray)
    print(io, Base.dims2string(size(A)), " ", nameof(typeof(A)), "{", eltype(A), ", ",
          indtype(A), "} with ", nnz(A), " stored ", nnz(A) == 1 ? "entry" : "entries",
          " in ", storage_name(A))
end

# the device-side representation, e.g. in an error message about a kernel argument, cannot
# be copied to the host
on_device(A) = A.nzVal isa AbstractDeviceArray

Base.print_array(io::IO, A::GPUSparseMatrix) =
    on_device(A) ? nothing : Base.print_array(io, SparseMatrixCSC(A))

Base.show(io::IO, A::GPUSparseMatrix) =
    on_device(A) ? summary(io, A) : show(io, SparseMatrixCSC(A))
Base.show(io::IO, x::GPUSparseVector) =
    on_device(x) ? summary(io, x) : show(io, SparseVector(x))
# (SparseArrays 1.10 has a method for this one)
Base.show(io::IOContext, x::GPUSparseVector) =
    on_device(x) ? summary(io, x) : show(io, SparseVector(x))

# SparseArrays prints the stored entries of a vector as part of its `show` method, below
# a summary that we replace by our own
function Base.show(io::IO, mime::MIME"text/plain", x::GPUSparseVector)
    summary(io, x)
    (nnz(x) == 0 || on_device(x)) && return
    println(io, ":")
    entries = sprint(show, mime, SparseVector(x); context=io)
    print(io, split(entries, '\n'; limit=2)[2])
end
