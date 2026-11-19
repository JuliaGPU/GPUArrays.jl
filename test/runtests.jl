using ParallelTestRunner: runtests, parse_args
using Test
import GPUArrays

include("testsuite.jl")

const init_worker_code = quote
    using Test, JLArrays, SparseArrays

    include("testsuite.jl")

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

# the sparse implementation should not add method ambiguities (the dense code has a few)
testsuite["sparse/ambiguities"] = quote
    import GPUArrays
    sparse_sources = (joinpath("src", "host", "sparse"), joinpath("JLArrays", "src", "sparse.jl"))
    ambiguities = filter(Test.detect_ambiguities(GPUArrays, JLArrays; recursive=true)) do (a, b)
        any(m -> any(src -> occursin(src, string(m.file)), sparse_sources), (a, b))
    end
    @test isempty(ambiguities)
end

# captured variables that get boxed (`Core.Box`) lose their type
if isdefined(Test, :detect_closure_boxes)
    testsuite["closure boxes"] = quote
        import GPUArrays
        @test Test.detect_closure_boxes(GPUArrays, GPUArrays.GPUArraysCore, JLArrays) == []
    end
end

runtests(GPUArrays, args; init_code, init_worker_code, testsuite)
