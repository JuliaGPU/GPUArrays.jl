# development often happens in lockstep with other packages,
# so check-out the master branch of those packages.
using Pkg
if haskey(ENV, "TRAVIS") || haskey(ENV, "APPVEYOR") || haskey(ENV, "GITLAB_CI")
    Pkg.add(PackageSpec(name="Adapt", rev="master"))
end

using GPUArrays, Test

@testset "JLArray" begin
    GPUArrays.test(JLArray)
end
