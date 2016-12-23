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

function Base.show(io::IO, mt::MIME"text/plain", A::GPUArray)
    ctxname = typeof(context(A)).name.name
    println(io, "GPUArray with ctx: $(context(A)): ")
    show(io, mt, Array(A))
end
function Base.showcompact(io::IO, mt::MIME"text/plain", A::GPUArray)
    showcompact(io, mt, Array(A))
end

function Base.similar(x::GPUArray)
    simbuff = similar(buffer(x))
    GPUArray(simbuff, size(x), context = context(x))
end
function Base.similar{T}(x::GPUArray, ::Type{T})
    simbuff = similar(buffer(x), T)
    GPUArray(simbuff, size(x), context = context(x))
end
function Base.similar{T, N}(x::GPUArray, ::Type{T}, sz::NTuple{N, Int})
    simbuff = similar(buffer(x), T, sz)
    GPUArray(simbuff, size(x), context = context(x))
end

#=
Host to Device data transfers
=#

# don't want to jump straight into refactor hell, so don't force GPU packges to inherit from GPUBuffer
function GPUArray(
        buffer#=::GPUBuffer=#, sz::Tuple;
        context::Context=current_context()
    )
    b, T, N = buffer, eltype(buffer), length(sz)
    GPUArray{T, N, typeof(b), typeof(context)}(buffer, sz, context)
end
function GPUArray{T, N}(
        ::Type{T}, sz::Vararg{Int, N};
        kw_args...
    )
    GPUArray(T, sz; kw_args...)
end
function GPUArray{T}(
        ::Type{T}, sz::Tuple;
        context::Context=current_context(), kw_args...
    )
    b = create_buffer(context, T, sz; kw_args...)
    GPUArray(b, sz, context=context)
end

function GPUArray{T, N}(
        host_array::AbstractArray{T, N};
        context::Context=current_context(), kw_args...
    )
    concrete_ha = convert(Array, host_array)
    gpu_array = GPUArray(T, size(concrete_ha), context=context)
    unsafe_copy!(gpu_array, concrete_ha)
    gpu_array
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
