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
using FillArrays

struct ArrayAdaptor{AT} end
Adapt.adapt_storage(::ArrayAdaptor{AT}, xs::AbstractArray) where {AT} = AT(xs)

function compare(f, AT::Type{<:AbstractGPUArray}, xs...; kwargs...)
    # copy on the CPU, adapt on the GPU, but keep Ref's
    cpu_in = map(x -> isa(x, Base.RefValue) ? x[] : deepcopy(x), xs)
    gpu_in = map(x -> isa(x, Base.RefValue) ? x[] : adapt(ArrayAdaptor{AT}(), x), xs)

    cpu_out = f(cpu_in...; kwargs...)
    gpu_out = f(gpu_in...; kwargs...)

    if cpu_out isa Tuple && gpu_out isa Tuple
        all(zip(cpu_out,gpu_out)) do (cpu, gpu)
            collect(cpu) ≈ collect(gpu)
        end
    else
        collect(cpu_out) ≈ collect(gpu_out)
    end
end

function compare(f, AT::Type{<:Array}, xs...; kwargs...)
    # no need to actually run this tests: we have nothing to compoare against,
    # and we'll run it on a CPU array anyhow when comparing to a GPU array.
    #
    # this method exists so that we can at least run the test suite with Array,
    # and make sure we cover other tests (that don't call `compare`) too.
    return true
end

function supported_eltypes()
    (Float32, Float64, Int32, Int64, ComplexF32, ComplexF64)
end

# list of tests
const tests = Dict()
macro testsuite(name, ex)
    safe_name = lowercase(replace(name, " "=>"_"))
    fn = Symbol("test_$(safe_name)")
    quote
        $(esc(fn))(AT) = $(esc(ex))(AT)

        @assert !haskey(tests, $name)
        tests[$name] = $fn
    end
end

include("testsuite/construction.jl")
include("testsuite/gpuinterface.jl")
include("testsuite/indexing.jl")
include("testsuite/io.jl")
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
