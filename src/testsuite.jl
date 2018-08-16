# Abstract test suite that can be used for all packages inheriting from GPUArray

module TestSuite

using GPUArrays
using GPUArrays: mapidx, gpu_sub2ind

using LinearAlgebra
using Random
using Test

using FFTW
using FillArrays
using StaticArrays

convert_array(f, x) = f(x)
convert_array(f, x::Base.RefValue) = x[]

function compare(f, Typ, xs...)
    cpu_in = convert_array.(copy, xs)
    gpu_in = convert_array.(Typ, xs)
    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)
    cpu_out ≈ Array(gpu_out)
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
include("testsuite/fft.jl")
include("testsuite/blas.jl")
include("testsuite/random.jl")

function supported_eltypes()
    (Float32, Float64, Int32, Int64, ComplexF32, ComplexF64)
end

export run_tests, supported_eltypes

end


"""
Runs the entire GPUArrays test suite on array type `Typ`
"""
function test(Typ)
    TestSuite.test_construction(Typ)
    TestSuite.test_gpuinterface(Typ)
    TestSuite.test_indexing(Typ)
    TestSuite.test_io(Typ)
    TestSuite.test_base(Typ)
    #TestSuite.test_vectors(Typ)
    TestSuite.test_mapreduce(Typ)
    TestSuite.test_broadcasting(Typ)
    TestSuite.test_linalg(Typ)
    TestSuite.test_fft(Typ)
    TestSuite.test_blas(Typ)
    TestSuite.test_random(Typ)
end
