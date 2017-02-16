module GLSLIntrinsics

prescripts = Dict(
    Float32 => "",
    Float64 => "d",
    Int => "i",
    Int32 => "i",
    UInt => "u",
    Bool => "b"
)


immutable GLArray{T, N} end


function glsl_hygiene(sym)
    # TODO unicode
    # TODO figure out what other things are not allowed
    # TODO startswith gl_, but allow variables that are actually valid inbuilds
    x = string(sym)
    # this seems pretty hacky! #TODO don't just ignore dots!!!!
    # this is only fine right now, because most opengl intrinsics broadcast anyways
    # but i'm sure this won't hold in general
    x = replace(x, ".", "")
    x = replace(x, "#", "__")
    x = replace(x, "!", "_bang")
    if x == "out"
        x = "_out"
    end
    if x == "in"
        x = "_in"
    end
    x
end

glsl_name(x) = Symbol(glsl_hygiene(_glsl_name(x)))

function _glsl_name(T)
    str = if isa(T, Expr) && T.head == :curly
        string(T, "_", join(T.args, "_"))
    elseif isa(T, Symbol)
        string(T)
    elseif isa(T, Type)
        str = string(T.name.name)
        if !isempty(T.parameters)
            str *= string("_", join(T.parameters, "_"))
        end
        str
    else
        error("Not transpilable: $(typeof(T))")
    end
    return str
end

function _glsl_name{T, N}(x::Type{GLArray{T, N}})
    if !(N in (1, 2, 3))
        # TODO, fake ND arrays with 1D array
        error("GPUArray can't have more than 3 dimensions for now")
    end
    sz = glsl_sizeof(T)
    len = glsl_length(T)
    "image$(N)D$(len)x$(sz)_bindless"
end
function _glsl_name{N, T}(::Type{NTuple{N, T}})
    string(prescripts[T], "vec", N)
end
function _glsl_name(::typeof(^))
    "pow"
end


function _glsl_name(x::Union{AbstractString, Symbol})
    x
end
_glsl_name(x::Type{Void}) = "void"
_glsl_name(x::Type{Float64}) = "float"
_glsl_name(x::Type{UInt}) = "uint"
_glsl_name(x::Type{Bool}) = "bool"

# TODO this will be annoying on 0.6
# _glsl_name(x::typeof(gli.:(*))) = "*"
# _glsl_name(x::typeof(gli.:(<=))) = "lessThanEqual"
# _glsl_name(x::typeof(gli.:(+))) = "+"

function _glsl_name{F <: Function}(f::Union{F, Type{F}})
    # Taken from base... #TODO make this more stable
    _glsl_name(F.name.mt.name)
end



# Number types
# Abstract types
# for now we use Int, more accurate would be Int32. But to make things simpler
# we rewrite Int to Int32 implicitely like this!
typealias int Int
# same goes for float
typealias float Float64

typealias uint UInt

const ints = (int, Int32, uint)
const floats = (Float32, float)
const numbers = (ints..., floats..., Bool)

const Ints = Union{ints...}
const Floats = Union{floats...}
const Numbers = Union{numbers...}

const functions = (
    +, -, *, /, ^,
    sin, tan, sqrt
)

const Functions = Union{map(typeof, functions)...}

_vecs = []
for i = 2:4, T in numbers
    nvec = NTuple{i, T}
    name = glsl_name(nvec)
    push!(_vecs, nvec)
    if !isdefined(name)
        @eval typealias $name $nvec
    end
end

const vecs = (_vecs...)
const Vecs = Union{vecs...}
const Types = Union{vecs..., numbers..., GLArray}

@noinline function ret{T}(::Type{T})::T
    unsafe_load(Ptr{T}(C_NULL))
end

# intrinsics not defined in Base need a function stub:
for i = 2:4
    @eval begin
        function (::Type{NTuple{$i, T}}){T <: Numbers, N, T2 <: Numbers}(x::NTuple{N, T2})
            ntuple(i-> T(x[i]), Val{$i})
        end
    end
end

imageStore{T}(x::GLArray{T, 1}, i::int, val::NTuple{4, T}) = nothing
imageStore{T}(x::GLArray{T, 2}, i::ivec2, val::NTuple{4, T}) = nothing

imageLoad{T}(x::GLArray{T, 1}, i::int) = ret(NTuple{4, T})
imageLoad{T}(x::GLArray{T, 2}, i::ivec2) = ret(NTuple{4, T})
imageSize{T, N}(x::GLArray{T, N}) = ret(NTuple{N, int})

function is_intrinsic{F <: Function}(f::F, types)
    t = (types.parameters...)
    F <: Functions && all(T-> T <: Types, t) ||
    (isdefined(GLSLIntrinsics, Symbol(f)) && length(methods(f, types)) == 1) # if any funtion stub matches
end

#######################################
# globals
const gl_GlobalInvocationID = uvec3((0,0,0))

end # end GLSLIntrinsics

if !isdefined(:gli)
const gli = GLSLIntrinsics
end

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
    getindex(x, (i, j))
end
function Base.getindex{T <: Number}(x::gli.GLArray{T, 2}, idx::gli.ivec2)
    gli.imageLoad(x, idx)[1]
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::T, i::Integer)
    gli.imageStore(x, i, (val, val, val, val))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, i::Integer, j::Integer)
    setindex!(x, (val, val, val, val), (i, j))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, idx::gli.ivec2)
    setindex!(x, (val, val, val, val), idx)
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::NTuple{4, T}, idx::gli.ivec2)
    gli.imageStore(x, idx, val)
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::NTuple{4, T}, i::Integer)
    gli.imageStore(x, i, val)
end

####################################
# Be a type pirate on 0.5!
# We shall turn this package into 0.6 only, but 0.6 is broken right now
# so that's why we need pirating!
if VERSION < v"0.6"
    Base.broadcast{N}(f, a::NTuple{N, Any}, b::NTuple{N, Any}) = map(f, a, b)
    Base.broadcast{N}(f, a::NTuple{N, Any}) = map(f, a)
    Base.:(.<=){N}(a::NTuple{N, Any}, b::NTuple{N, Any}) = map(<=, a, b)
    Base.:(.*){N}(a::NTuple{N, Any}, b::NTuple{N, Any}) = map(*, a, b)
    Base.:(.+){N}(a::NTuple{N, Any}, b::NTuple{N, Any}) = map(+, a, b)
end
