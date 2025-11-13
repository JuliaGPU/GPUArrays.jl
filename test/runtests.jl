using ParallelTestRunner: runtests, parse_args
import GPUArrays

include("testsuite.jl")

const init_code = quote
    using Test, JLArrays, SparseArrays

    include("testsuite.jl")

    TestSuite.sparse_types(::Type{<:JLArray}) = (JLSparseVector, JLSparseMatrixCSC, JLSparseMatrixCSR)
    TestSuite.sparse_types(::Type{<:Array})   = (SparseVector, SparseMatrixCSC)

    # Disable Float16-related tests until JuliaGPU/KernelAbstractions#600 is resolved
    if isdefined(JLArrays.KernelAbstractions, :POCL)
        TestSuite.supported_eltypes(::Type{<:JLArray}) =
            setdiff(TestSuite.supported_eltypes(), [Float16, ComplexF16])
    end
end

args = parse_args(ARGS)

testsuite = Dict{String, Expr}()
for AT in (:JLArray, :Array), name in keys(TestSuite.tests)
    testsuite["$(AT)/$name"] = :(TestSuite.tests[$name]($AT))
end

runtests(GPUArrays, ARGS; init_code, testsuite)
