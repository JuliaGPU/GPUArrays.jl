type GPUArray{T, N, B, C} <: DenseArray{T, N}
  buffer::B
  size::NTuple{N, Int}
  context::C
end

#=
context interface
=#

buffer(A::GPUArray) = A.buffer
context(A::GPUArray) = A.context

#=
AbstractArray interface
=#

Base.eltype{B, T, N, C}(::Type{GPUArray{T, N, B, C}}) = T
Base.eltype{T}(::GPUArray{T}) = T
Base.size(A::GPUArray) = A.size
Base.size(A::GPUArray, i::Int) = A.size[i]

function Base.show(io::IO, A::GPUArray)
    ctxname = typeof(context(A)).name.name
    println(io, "GPUArray with ctx: $(context(A)): ")
    show(io, Array(A))
end
function Base.showarray(io::IO, A::GPUArray)
    ctxname = typeof(context(A)).name.name
    println(io, "GPUArray with ctx: $(context(A)): ")
    show(io, Array(A))
end
function Base.showarray(io::IO, mt::MIME"text/plain", A::GPUArray)
    ctxname = typeof(context(A)).name.name
    println(io, "GPUArray with ctx: $(context(A)): ")
    show(io, mt, Array(A))
end
function Base.display(A::GPUArray)
    display("text/plain", "GPUArray($(context(A)))")
    display(Array(A))
end
#=
Host to Device data transfers
=#

# don't want to jump straight into refactor hell, so don't force GPU packges to inherit from GPUBuffer
function GPUArray(buffer#=::GPUBuffer=#, sz::Tuple, ctx::Context=current_context())
    b, T, N = buffer, eltype(buffer), length(sz)
    GPUArray{T, N, typeof(b), typeof(ctx)}(buffer, sz, ctx)
end

function GPUArray{T, N}(host_array::AbstractArray{T, N}, ctx::Context=current_context(); kw_args...)
    concrete_ha = convert(Array, host_array)
    b = create_buffer(ctx, concrete_ha; kw_args...)
    GPUArray{T, N, typeof(b), typeof(ctx)}(
        b, size(concrete_ha), ctx
    )
end

#=
Device to host data transfers
=#

function (::Type{Array}){T, N}(device_array::GPUArray{T, N})
    Array{T, N}(device_array)
end
function (AT::Type{Array{T, N}}){T, N}(device_array::GPUArray)
    convert(AT, Array(device_array))
end
function (AT::Type{Array{T, N}}){T, N}(device_array::GPUArray{T, N})
    hostarray = similar(AT, size(device_array))
    unsafe_copy!(hostarray, device_array)
    hostarray
end


#=
Copying
=#

function Base.unsafe_copy!{T, N}(dest::Array{T, N}, source::GPUArray{T, N})
    Base.unsafe_copy!(dest, buffer(source))
end
function Base.unsafe_copy!{T, N}(dest::GPUArray{T, N}, source::Array{T, N})
    Base.unsafe_copy!(buffer(dest), source)
end

export buffer, context
