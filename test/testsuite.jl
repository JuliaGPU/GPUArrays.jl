# test suite that can be used for all packages inheriting from GPUArrays
#
# use this by either calling `test`, or inspecting the dictionary `tests`

module TestSuite

export supported_eltypes

using GPUArrays

using LinearAlgebra
using Random
using Test

using Adapt

struct ArrayAdaptor{AT} end
Adapt.adapt_storage(::ArrayAdaptor{AT}, xs::AbstractArray) where {AT} = AT(xs)

test_result(a::Number, b::Number) = a ≈ b
function test_result(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:Number}
    collect(a) ≈ collect(b)
end
function test_result(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:NTuple{N,<:Number} where {N}}
    ET = eltype(T)
    reinterpret(ET, collect(a)) ≈ reinterpret(ET, collect(b))
end
function test_result(as::NTuple{N,Any}, bs::NTuple{N,Any}) where {N}
    all(zip(as, bs)) do (a, b)
        test_result(a, b)
    end
end

function compare(f, AT::Type{<:AbstractGPUArray}, xs...; kwargs...)
    # copy on the CPU, adapt on the GPU, but keep Ref's
    cpu_in = map(x -> isa(x, Base.RefValue) ? x[] : deepcopy(x), xs)
    gpu_in = map(x -> isa(x, Base.RefValue) ? x[] : adapt(ArrayAdaptor{AT}(), x), xs)

    cpu_out = f(cpu_in...; kwargs...)
    gpu_out = f(gpu_in...; kwargs...)

    test_result(cpu_out, gpu_out)
end

function compare(f, AT::Type{<:Array}, xs...; kwargs...)
    # no need to actually run this tests: we have nothing to compoare against,
    # and we'll run it on a CPU array anyhow when comparing to a GPU array.
    #
    # this method exists so that we can at least run the test suite with Array,
    # and make sure we cover other tests (that don't call `compare`) too.
    return true
end

# element types that are supported by the array type
supported_eltypes(AT, test) = supported_eltypes(AT)
supported_eltypes(AT) = supported_eltypes()
supported_eltypes() = (Int16, Int32, Int64,
                       Float16, Float32, Float64,
                       ComplexF16, ComplexF32, ComplexF64,
                       Complex{Int16}, Complex{Int32}, Complex{Int64})

# some convenience predicates for filtering test eltypes
isrealtype(T) = T <: Real
iscomplextype(T) = T <: Complex
isrealfloattype(T) = T <: AbstractFloat
isfloattype(T) = T <: AbstractFloat || T <: Complex{<:AbstractFloat}

# list of tests
const tests = Dict()
macro testsuite(name, ex)
    safe_name = lowercase(replace(replace(name, " "=>"_"), "/"=>"_"))
    fn = Symbol("test_$(safe_name)")
    quote
        # the supported element types can be overrided by passing in a different set,
        # or by specializing the `supported_eltypes` function on the array type and test.
        $(esc(fn))(AT; eltypes=supported_eltypes(AT, $(esc(fn)))) = $(esc(ex))(AT, eltypes)

        @assert !haskey(tests, $name)
        tests[$name] = $fn
    end
end

include("testsuite/construction.jl")
include("testsuite/gpuinterface.jl")
include("testsuite/indexing.jl")
include("testsuite/base.jl")
#include("testsuite/vector.jl")
include("testsuite/reductions.jl")
include("testsuite/broadcasting.jl")
include("testsuite/linalg.jl")
include("testsuite/math.jl")
include("testsuite/random.jl")
include("testsuite/uniformscaling.jl")
include("testsuite/statistics.jl")

"""
Runs the entire GPUArrays test suite on array type `AT`
"""
function test(AT::Type)
    for (name, fun) in tests
        code = quote
            $fun($AT)
        end
        @eval @testset $name $code
    end
end

end
