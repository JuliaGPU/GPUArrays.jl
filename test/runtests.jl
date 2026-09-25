using ParallelTestRunner: runtests, parse_args
import GPUArrays

include("testsuite.jl")

const init_worker_code = quote
    using Test, JLArrays, SparseArrays

    include("testsuite.jl")
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

# the sparse implementation should not add method ambiguities (the dense code has a few)
testsuite["sparse/ambiguities"] = quote
    import GPUArrays
    sparse_sources = (joinpath("src", "host", "sparse"), joinpath("JLArrays", "src", "sparse.jl"),
                      joinpath("JLArrays", "src", "algorithms.jl"))
    ambiguities = filter(Test.detect_ambiguities(GPUArrays, JLArrays; recursive=true)) do (a, b)
        any(m -> any(src -> occursin(src, string(m.file)), sparse_sources), (a, b))
    end
    @test isempty(ambiguities)
end

runtests(GPUArrays, args; init_code, init_worker_code, testsuite)
