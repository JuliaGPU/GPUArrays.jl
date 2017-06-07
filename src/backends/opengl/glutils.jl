
"""
Statically sized uniform buffer.
Supports push!, but with fixed memory, so it will error after reaching
it's preallocated length.
"""
type UniformBuffer{T, N}
    buffer::GLBuffer{T}
    offsets::NTuple{N, Int}
    elementsize::Int
    length::Int
end
const GLSLScalarTypes = Union{Float32, Int32, UInt32}
Base.eltype{T, N}(::UniformBuffer{T, N}) = T


function glsl_sizeof(T)
    T <: Bool && return sizeof(Int32)
    T <: GLSLScalarTypes && return sizeof(T)
    # TODO Propper translation and sizes!
    T <: Function && return sizeof(Vec4f0) # sizeof(EmptyStruct) padded to Vec4f0
    ET = eltype(T)
    if T <: Mat
        return sizeof(ET) * 4 * size(T, 2)
    end
    # must be vector like #TODO assert without restricting things too much
    N = length(T)
    @assert N <= 4
    N <= 2 && return 2 * sizeof(ET)
    return 4 * sizeof(ET)
end

function std140_offsets{T}(::Type{T})
    elementsize = 0
    offsets = if T <: GLSLScalarTypes
        elementsize = sizeof(T)
        (0,)
    else
        offset = 0
        offsets = ntuple(nfields(T)) do i
            ft = fieldtype(T, i)
            sz = glsl_sizeof(ft)
            of = offset
            offset += sz
            of
        end
        elementsize = offset
        offsets
    end
    offsets, elementsize
end

"""
    Pre allocates an empty buffer with `max_batch_size` size
    which can be used to store multiple uniform blocks of type T
"""
function UniformBuffer{T}(::Type{T}, max_batch_size = 1024, mode = GL_STATIC_DRAW)
    offsets, elementsize = std140_offsets(T)
    buffer = GLBuffer{T}(
        max_batch_size,
        elementsize * max_batch_size,
        GL_UNIFORM_BUFFER, mode
    )
    UniformBuffer(buffer, offsets, elementsize, 0)
end

"""
    Creates an Uniform buffer with the contents of `data`
"""
function UniformBuffer{T}(data::T, mode = GL_STATIC_DRAW)
    buffer = UniformBuffer(T, 1, mode)
    push!(buffer, data)
    buffer
end

function assert_blocksize(buffer::UniformBuffer, program, blockname::String)
    block_index = glGetUniformBlockIndex(program, blockname)
    blocksize_ref = Ref{GLint}(0)
    glGetActiveUniformBlockiv(
        program, block_index,
        GL_UNIFORM_BLOCK_DATA_SIZE, blocksize_ref
    )
    blocksize = blocksize_ref[]
    @assert buffer.elementsize * length(buffer.buffer) == blocksize
end

_getfield(x::GLSLScalarTypes, i) = x
_getfield(x, i) = getfield(x, i)

function iterate_fields{T, N}(buffer::UniformBuffer{T, N}, x, index)
    offset = buffer.elementsize * (index - 1)
    x_ref = isimmutable(x) ? Ref(x) : x
    base_ptr = pointer_from_objref(x_ref)
    ntuple(Val{N}) do i
        offset + buffer.offsets[i], base_ptr + fieldoffset(T, i), sizeof(fieldtype(T, i))
    end
end

function Base.setindex!{T, N}(buffer::UniformBuffer{T, N}, element::T, idx::Integer)
    if idx > length(buffer.buffer)
        throw(BoundsError(buffer, idx))
    end
    GLAbstraction.bind(buffer.buffer)
    for (offset, ptr, size) in iterate_fields(buffer, element, idx)
        glBufferSubData(GL_UNIFORM_BUFFER, offset, size, ptr)
    end
    GLAbstraction.bind(buffer.buffer, 0)
    element
end

function Base.push!{T, N}(buffer::UniformBuffer{T, N}, element::T)
    buffer.length += 1
    buffer[buffer.length] = element
    buffer
end


function check_copy_bounds(
        dest, d_offset::Integer,
        src, s_offset::Integer,
        amount::Integer
    )
    amount > 0 || throw(ArgumentError(string("tried to copy n=", amount, " elements, but amount should be nonnegative")))
    if s_offset < 1 || d_offset < 1 ||
            s_offset + amount - 1 > length(src) ||
            d_offset + amount - 1 > length(dest)
        throw(BoundsError())
    end
    nothing
end


function copy!{T}(
        dest::gl.GLBuffer{T}, d_range::CartesianRange{CartesianIndex{1}},
        src::Vector{T}, s_range::CartesianRange{CartesianIndex{1}},
    )
    amount = length(d_range)
    if length(s_range) != amount
        throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
    end
    amount == 0 && return dest
    d_offset = first(d_range)[1]
    s_offset = first(s_range)[1]
    check_copy_bounds(dest, d_offset, src, s_offset, amount)
    multiplicator = sizeof(T)
    nsz = multiplicator * amount
    bind(dest)
    glBufferSubData(dest.buffertype, multiplicator * (d_offset - 1), nsz, Ref(src, s_offset))
    bind(dest, 0)
end

function copy!{T}(
        dest::Vector{T}, d_range::CartesianRange{CartesianIndex{1}},
        src::gl.GLBuffer{T}, s_range::CartesianRange{CartesianIndex{1}},
    )
    amount = length(d_range)
    if length(s_range) != amount
        throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
    end
    amount == 0 && return dest
    d_offset = first(d_range)[1]
    s_offset = first(s_range)[1]
    check_copy_bounds(dest, d_offset, src, s_offset, amount)
    multiplicator = sizeof(T)
    nsz = multiplicator * amount
    bind(src)
    glGetBufferSubData(
        src.buffertype, multiplicator * (s_offset - 1), nsz,
        Ref(dest, d_offset)
    )
    bind(src, 0)
    dest
end


# copy between two buffers
function copy!{T}(
        dest::gl.GLBuffer{T}, d_range::CartesianRange{CartesianIndex{1}},
        src::gl.GLBuffer{T}, s_range::CartesianRange{CartesianIndex{1}}
    )
    amount = length(d_range)
    if length(s_range) != amount
        throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
    end
    amount == 0 && return dest
    d_offset = first(d_range)[1]
    s_offset = first(s_range)[1]
    check_copy_bounds(dest, d_offset, src, s_offset, amount)
    multiplicator = sizeof(T)
    nsz = multiplicator * amount

    glBindBuffer(GL_COPY_READ_BUFFER, src.id)
    glBindBuffer(GL_COPY_WRITE_BUFFER, dest.id)
    glCopyBufferSubData(
        GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
        multiplicator * (s_offset - 1),
        multiplicator * (d_offset - 1),
        multiplicator * amount
    )
    glBindBuffer(GL_COPY_READ_BUFFER, 0)
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0)
    return nothing
end
