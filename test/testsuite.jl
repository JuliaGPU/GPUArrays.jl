# test suite that can be used for all packages inheriting from GPUArrays

module TestSuite

export supported_eltypes

using GPUArrays

using LinearAlgebra
using Random
using Test

using FFTW
using FillArrays

convert_array(f, x) = f(x)
convert_array(f, x::Base.RefValue) = x[]

function compare(f, AT::Type{<:AbstractGPUArray}, xs...; kwargs...)
    cpu_in = convert_array.(copy, xs)
    gpu_in = convert_array.(AT, xs)
    cpu_out = f(cpu_in...; kwargs...)
    gpu_out = f(gpu_in...; kwargs...)
    collect(cpu_out) ≈ collect(gpu_out)
end

function supported_eltypes()
    (Float32, Float64, Int32, Int64, ComplexF32, ComplexF64)
end

include("testsuite/construction.jl")
include("testsuite/gpuinterface.jl")
include("testsuite/indexing.jl")
include("testsuite/io.jl")
include("testsuite/base.jl")
include("testsuite/vector.jl")
include("testsuite/mapreduce.jl")
include("testsuite/broadcasting.jl")
include("testsuite/linalg.jl")
include("testsuite/math.jl")
include("testsuite/fft.jl")
include("testsuite/random.jl")


"""
Runs the entire GPUArrays test suite on array type `AT`
"""
function test(AT::Type{<:AbstractGPUArray})
    TestSuite.test_construction(AT)
    TestSuite.test_gpuinterface(AT)
    TestSuite.test_indexing(AT)
    TestSuite.test_io(AT)
    TestSuite.test_base(AT)
    TestSuite.test_mapreduce(AT)
    TestSuite.test_broadcasting(AT)
    TestSuite.test_linalg(AT)
    TestSuite.test_math(AT)
    TestSuite.test_fft(AT)
    TestSuite.test_random(AT)
end

end
