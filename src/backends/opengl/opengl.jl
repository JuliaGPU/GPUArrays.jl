module GLBackend

using ..GPUArrays

import GPUArrays: buffer, create_buffer, acc_broadcast!
import GPUArrays: Context, GPUArray, context, broadcast_index

import GLAbstraction, GLWindow, GLFW
using ModernGL
const gl = GLAbstraction

include("compilation.jl")

immutable GLContext <: Context
    window::GLFW.Window
end
Base.show(io::IO, ctx::GLContext) = print(io, "GLContext")

typealias GLArrayBuff{T, N} GPUArray{T, N, gl.GLBuffer{T}, GLContext}
typealias GLArrayTex{T, N} GPUArray{T, N, gl.Texture{T, N}, GLContext}
typealias GLArray{T, N} Union{GLArrayBuff{T, N}, GLArrayTex{T, N}}


function any_context()
    window = GLWindow.create_glcontext()
    GLFW.HideWindow(window)
    GLContext(window)
end

global all_contexts, current_context, init
let contexts = GLContext[]
    all_contexts() = copy(contexts)::Vector{GLContext}
    current_context() = last(contexts)::GLContext
    function init(; ctx = any_context())
        init(ctx)
    end
    init(ctx::GLWindow.Screen) = init(GLContext(GLWindow.nativewindow(ctx)))
    function init(ctx::GLContext)
        GPUArrays.make_current(ctx)
        push!(contexts, ctx)
        ctx
    end
end

function create_buffer{T,N}(ctx::GLContext, ::Type{T}, sz::NTuple{N, Int}; kw_args...)
    gl.Texture(T, sz; kw_args...)
end

function glTexSubImage{N}(tex, offset::NTuple{N, Int}, width::NTuple{N, Int}, data)
    glfun = N == 1 ? glTexSubImage1D : N == 2 ? glTexSubImage2D : N==3 ? glTexSubImage3D : error("Dim $N not supported")
    glfun(tex.texturetype, 0, offset..., width..., tex.format, tex.pixeltype, data)
end
function Base.unsafe_copy!{ET, ND}(
        dest::gl.Texture{ET, ND},
        source::Array{ET, ND}
    )
    gl.update!(dest, source)
    nothing
end
function Base.unsafe_copy!{ET, ND}(
        dest::gl.Texture{ET, ND},
        source::gl.GLBuffer{ET},
        offset, widths
    )
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, source.id)
    # Select the appropriate texture
    gl.bind(dest)
    glTexSubImage(dest, offset, widths, C_NULL)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    gl.bind(dest, 0)
    nothing
end

function Base.unsafe_copy!{ET, ND}(
        dest::GLArrayTex{ET, ND},
        source::GLArrayBuff{ET, ND}
    )
    unsafe_copy!(buffer(dest), buffer(source), ntuple(x->0, ND), size(source))
end

function Base.convert{ET, ND}(
        ::Type{GLArrayTex{ET, ND}},
        A::GLArrayBuff{ET, ND}
    )
    texB = GPUArray(gl.Texture(ET, size(A)), size(A), context(A))
    unsafe_copy!(texB, A)
    texB
end


################################################################################
# Broadcast

#Broadcast
@generated function broadcast_index{T, N}(arg::gli.GLArray{T, N}, shape, idx)
    iexpr = Expr(:tuple)
    for i = 1:N
        push!(iexpr.args, :(sz[$i] <= shape[$i] ? idx[$i] : 1))
    end
    quote
        sz = size(arg)
        arg[$iexpr]
    end
end
broadcast_index(arg, shape, idx) = arg


for N = 1:3
    for i=0:6
        args = ntuple(x-> Symbol("arg_", x), i)
        fargs = ntuple(x-> :(broadcast_index($(args[x]), sz, idx)), i)
        @eval begin
            function broadcast_kernel{T}(A::gli.GLArray{T, $N}, f, $(args...))
                idx =  NTuple{2, Int}(GlobalInvocationID())
                sz  = size(A)
                A[idx] = f($(fargs...))
                return
            end
        end
    end
end


size3d{T}(A::GLArrayTex{T, 1}) = (size(A, 1), 1, 1)
size3d{T}(A::GLArrayTex{T, 2}) = (size(A)..., 1)
size3d{T}(A::GLArrayTex{T, 3}) = size(A)

function acc_broadcast!{F <: Function, T, N}(f::F, A::GLArrayTex{T, N}, args::Tuple)
    glfunc = ComputeProgram(broadcast_kernel, (A, f, args...))
    glfunc((A, f, args...), size3d(A))
end


# TODO We tread Float64 as the default and map it to Float32 in glsl. HACK ALERT
# We mainly do this, because of 1.0 being Float64 as default
to_glsl_types(::Type{Float32}) = Float64
to_glsl_types{T}(arg::T) = to_glsl_types(T)
to_glsl_types{T}(::Type{T}) = T
function to_glsl_types{T <: GPUArrays.AbstractAccArray}(arg::T)
    if isa(buffer(arg), gl.Texture)
        et = to_glsl_types(eltype(arg))
        return gli.GLArray{et, ndims(arg)}
    else
        error("Not implemented yet: $T")
        return typeof(arg)
    end
end
function to_glsl_types(args::Union{Vector, Tuple})
    map(to_glsl_types, args)
end

function bindlocation{T, N}(A::GLArrayTex{T, N}, i)
    t = buffer(A)
    glBindImageTexture(i, t.id, 0, GL_FALSE, 0, GL_READ_WRITE, t.internalformat)
end

# TODO integrate buffers
# function bind(t::GLBuffer, i)
#     glBindImageTexture(i, t.id, 0, GL_FALSE, 0, GL_READ_WRITE, t.internalformat)
# end

function bindlocation(t, i)
    gluniform(i, t)
end
#Functions will be declared in the shader as a constant, so we don't need to bind them
function bindlocation(t::Function, i)
end


# implement BLAS backend

function blas_module(A::GLContext)
    if blas_supported(CLContext)
        CUBLAS
    elseif blas_supported(CUContext)
        CUBLAS
    else
        BLAS # performance error ?!
    end
end


end
