# GPUArrays development often happens in lockstep with other packages, so try to match branches
using Pkg
function match_package(package, var)
    try
        branch = ENV[var]
        Pkg.add(PackageSpec(name=package, rev=branch))
        @info "Installed $package from branch $branch"
    catch ex
        @warn "Could not install $package from $branch branch, trying master" exception=ex
        Pkg.add(PackageSpec(name=package, rev="master"))
        @info "Installed $package from master"
    end
end
haskey(ENV, "TRAVIS")    && match_package("Adapt", "TRAVIS_PULL_REQUEST_BRANCH")
haskey(ENV, "APPVEYOR")  && match_package("Adapt", "APPVEYOR_PULL_REQUEST_HEAD_REPO_BRANCH")
haskey(ENV, "GITLAB_CI") && match_package("Adapt", "CI_COMMIT_REF_NAME")

using GPUArrays, Test

@testset "JLArray" begin
    GPUArrays.test(JLArray)
end

if haskey(ENV, "GITLAB_CI")
    match_package("CuArrays", "CI_COMMIT_REF_NAME")
    @testset "CuArray" begin
        Pkg.test("CuArrays")
    end
end
