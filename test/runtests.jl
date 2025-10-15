using ParallelTestRunner: runtests
import GPUArrays

include("testsuite.jl")

const init_code = quote
    using Test, JLArrays

    include("testsuite.jl")

    # Disable Float16-related tests until JuliaGPU/KernelAbstractions#600 is resolved
    if isdefined(JLArrays.KernelAbstractions, :POCL)
        TestSuite.supported_eltypes(::Type{<:JLArray}) =
            setdiff(TestSuite.supported_eltypes(), [Float16, ComplexF16])
    end
end

custom_tests = Dict{String, Expr}()
for AT in (:JLArray, :Array), name in keys(TestSuite.tests)
    custom_tests["$(AT)/$name"] = :(TestSuite.tests[$name]($AT))
end

function test_filter(test)
    if startswith(test, "testsuite")
        return false
    end
    return true
end

runtests(GPUArrays, ARGS; init_code, custom_tests, test_filter)
