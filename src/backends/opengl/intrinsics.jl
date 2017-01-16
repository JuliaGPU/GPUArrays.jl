# Pirate a type or function. This is needed, when an existing function/type
# needs to be overwritten as an GLSL intrinsic. This only works if the
# function has exactly the same signature.
# usage: @pirate TargetType glsl_type or function
# this will trigger the transpiler to replace that function type with the corresponding intrinsic
# The results will be placed in pirate loot, which resolves a name to it's signatures
# and from the signature to the glsl couterpart
pirate_loot = Dict{Any, Any}(
# TODO figure out if we should always use Float32 and default literals like 1.0 to 32bit
# this seems to be annoying though, since it'll make it hard to use Float64
    :Float32 => :float,
    :Float64 => :double,
    # We only ever translate to int32 in OpenGL, since 64 seem to be non existant
    # TODO does it really have to be like this?
    :Int32 => :int,
    :Int64 => :int,
    :^ => [(Number, Number) => :pow],
    Vec{2, Int} => [
        Tuple{Vec{3, Int}} => :ivec2,
        Tuple{Vec{2, Bool}} => :ivec2,
    ],
    GlobalRef(Main, :Vec) => [Tuple{Float64, Float64, Float64, Float64} => :vec4],
    :getindex => [Tuple{Vec, Integer} => :getindex]
    #:(Vec{2, Int}) => :ivec2
)



baremodule GLSLIntrinsics
# GLSL heavily relies on fixed size vector operation, which is why a lot of
# intrinsics need fixed size vectors.
import Base
import GeometryTypes
import GeometryTypes: Vec
import Base: Symbol, string, @noinline, Ptr, C_NULL, unsafe_load, @eval

# Number types
import Base: Float64, Float32, Int64, Int32
# Abstract types
const Floats = Union{Float64, Float32}
const Ints = Union{Int64, Int32}
const Numbers = Union{Floats, Ints}


# helper functions to trick inference into believing
# ret is returning a value of type T. Must obviously not be called!
@noinline function ret{T}(::Type{T})::T
    unsafe_load(Ptr{T}(C_NULL))
end

immutable GLArray{T, N} end


imageStore{T}(x::GLArray{T, 1}, i::Integer, val::Vec{4, T}) = nothing
imageStore{T, I <: Integer}(x::GLArray{T, 2}, i::Vec{2, I}, val::Vec{4, T}) = nothing

imageLoad{T}(x::GLArray{T, 1}, i::Integer) = ret(Vec{4, T})
imageLoad{T, I <: Integer}(x::GLArray{T, 2}, i::Vec{2, I}) = ret(Vec{4, T})

cos{T <: Floats}(x::T) = ret(T)
sin{T <: Floats}(x::T) = ret(T)
sqrt{T <: Floats}(x::T) = ret(T)

imageSize{T, N}(x::GLArray{T, N}) = ret(Vec{N, Int})

(<=){N, T <: Numbers}(x::Vec{N, T}, y::Vec{N, T}) = ret(Vec{N, Bool})
(*){N, T <: Numbers}(x::Vec{N, Bool}, y::Vec{N, T}) = ret(Vec{N, T})
(*){N, T <: Numbers}(x::Vec{N, T}, y::Vec{N, Bool}) = ret(Vec{N, T})
(*){N, T <: Numbers}(x::Vec{N, T}, y::Vec{N, T}) = ret(Vec{N, T})
(+){N, T <: Numbers}(x::Vec{N, T}, y::Vec{N, T}) = ret(Vec{N, T})


=={T <: Numbers}(x::T, y::T) = false
+{T <: Numbers}(x::T, y::T) = ret(T)
*{T <: Numbers}(x::T, y::T) = ret(T)
/{T <: Numbers}(x::T, y::T) = ret(Base.promote_op(/, T, T))


#######################################
# type constructors

# vecs
for n in (2,3,4), (T, ps) in ((Float64, ""), (Int, "i"))
    name = Symbol(string(ps, "vec", n))
    @eval $name(x::$T, y::$T) = ret(Vec{$n, $T})

end
ivec2(x::Vec{3, Int}) = ret(Vec{2, Int})
ivec2(x::Vec{3, UInt}) = ret(Vec{2, Int})
ivec2(x::Vec{2, Int}) = ret(Vec{2, Int})
ivec2(x::Vec{2, UInt}) = ret(Vec{2, Int})
ivec2(x::Vec{2, Bool}) = ret(Vec{2, Int})


#######################################
# globals

const gl_GlobalInvocationID = Vec{3, UInt}(0,0,0)

end # end GLSLIntrinsics

import .GLSLIntrinsics

const gli = GLSLIntrinsics

function GlobalInvocationID()
    gli.gl_GlobalInvocationID
end

function Base.size{T, N}(x::gli.GLArray{T, N})
    gli.imageSize(x)
end
function Base.getindex{T}(x::gli.GLArray{T, 1}, i::Integer)
    gli.imageLoad(x, i)
end
function Base.getindex{T}(x::gli.GLArray{T, 2}, i::Integer, j::Integer)
    getindex(x, Vec(i, j))
end
function Base.getindex{T <: Number}(x::gli.GLArray{T, 2}, idx::Vec{2, Int})
    gli.imageLoad(x, idx)[1]
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::T, i::Integer)
    gli.imageStore(x, i, Vec(val, val, val, val))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, i::Integer, j::Integer)
    setindex!(x, Vec(val, val, val, val), Vec(i, j))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, idx::Vec{2, Int})
    setindex!(x, Vec(val, val, val, val), idx)
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::Vec{4, T}, idx::Vec{2, Int})
    gli.imageStore(x, idx, val)
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::Vec{4, T}, i::Integer)
    gli.imageStore(x, i, val)
end
