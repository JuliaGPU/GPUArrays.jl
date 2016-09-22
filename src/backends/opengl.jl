module GLBackend

using ..GPUArrays
import GPUArrays: buffer, create_buffer, Context, GPUArray, context

import GLAbstraction, GLWindow, GLFW
using ModernGL
const gl = GLAbstraction

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
    function init(; ctx=any_context())
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
    gl.GLBuffer(T, prod(sz); kw_args...)
end

function glTexSubImage{N}(tex, offset::NTuple{N, Int}, width::NTuple{N, Int}, data)
    glfun = N == 1 ? glTexSubImage1D : N == 2 ? glTexSubImage2D : N==3 ? glTexSubImage3D : error("Dim $N not supported")
    glfun(tex.texturetype, 0, offset..., width..., tex.format, tex.pixeltype, data)
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

end
