using GPUArrays, Test, Pkg

include("testsuite.jl")

@testset "JLArray" begin
    # install the JLArrays subpackage in a temporary environment
    old_project = Base.active_project()
    Pkg.activate(; temp=true)
    Pkg.develop(path=joinpath(dirname(@__DIR__), "lib", "JLArrays"))

    using JLArrays

    jl([1])

    TestSuite.test(JLArray)

    Pkg.activate(old_project)
end

#=
@testset "JLArray" begin
    using JLArrays

    jl([1])

    TestSuite.test(JLArray)
end
=#

@testset "Array" begin
    TestSuite.test(Array)
end
