using ParallelTestRunner: runtests, parse_args
using Test
import GPUArrays

include("testsuite.jl")

const init_worker_code = quote
    using Test, JLArrays, SparseArrays

    include("testsuite.jl")

    TestSuite.sparse_types(::Type{<:JLArray}) = (JLSparseVector, JLSparseMatrixCSC, JLSparseMatrixCSR)
    TestSuite.sparse_types(::Type{<:Array})   = (SparseVector, SparseMatrixCSC)

    # JLArray kernels run on POCL, whose SPIR-V back-end lacks 128-bit integers
    TestSuite.supported_eltypes(::Type{<:JLArray}) =
        filter(!=(Int128), TestSuite.supported_eltypes())
end

const init_code = quote
    using Test, JLArrays, SparseArrays

    import ..TestSuite
end

args = parse_args(ARGS)

testsuite = Dict{String, Expr}()
for AT in (:JLArray, :Array), name in keys(TestSuite.tests)
    testsuite["$(AT)/$name"] = :(TestSuite.tests[$name]($AT))
end

testsuite["JLArray/kernelabstractions"] = quote
    import KernelAbstractions
    import KernelAbstractions as KA
    using JLArrays.JLKernels

    include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

    # JLArrays don't convert dense arrays to sparse ones, and the other two tests only run
    # kernels on KernelAbstractions' own CPU back end
    skip_tests = Set(["sparse", "CPU synchronization", "fallback test: callable types"])
    # JLArray kernels are compiled by POCL, like the CPU back end's, so check them the same way
    Testsuite.testsuite(JLBackend, "CPU", KernelAbstractions.POCL, JLArray,
                        KernelAbstractions.POCL.CLDeviceArray; skip_tests)
end

testsuite["JLArray/kernelinterface"] = quote
    import KernelInterface
    import KernelInterface as KI
    using JLArrays.JLKernels
    import Adapt

    include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

    Testsuite.testsuite(JLBackend(), JLArray)
end

# captured variables that get boxed (`Core.Box`) lose their type
if isdefined(Test, :detect_closure_boxes)
    testsuite["closure boxes"] = quote
        import GPUArrays
        @test Test.detect_closure_boxes(GPUArrays, GPUArrays.GPUArraysCore, JLArrays) == []
    end
end

# a type without a COO counterpart must hit a MethodError, not recurse (#780)
testsuite["sparse/coo_type"] = quote
    import GPUArrays
    @test_throws MethodError GPUArrays.coo_type(sparse([1], [1], [1.0], 2, 2))
end

runtests(GPUArrays, args; init_code, init_worker_code, testsuite)
