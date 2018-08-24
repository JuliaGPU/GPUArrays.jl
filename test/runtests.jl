using GPUArrays, Test
using GPUArrays.TestSuite

using Pkg

@testset "JLArray" begin
    GPUArrays.test(JLArray)
end

function test_package(package, branch=nothing)
    mktempdir() do devdir
        withenv("JULIA_PKG_DEVDIR" => devdir) do
            # try to install from the same branch of GPUArrays
            try
                if branch === nothing
                    branch = chomp(read(`git -C $(@__DIR__) rev-parse --abbrev-ref HEAD`, String))
                    branch == "HEAD" && error("in detached HEAD state")
                end
                Pkg.add(PackageSpec(name=package, rev=String(branch)))
                @info "Installed $package from $branch branch"
            catch ex
                @warn "Could not install $package from same branch as GPUArrays, trying master branch" exception=ex
                Pkg.add(PackageSpec(name=package, rev="master"))
            end

            Pkg.test(package)
        end
    end
end

if haskey(ENV, "GITLAB_CI")
    branch = ENV["CI_COMMIT_REF_NAME"]
    @testset "CuArray" begin
        test_package("CuArrays", branch)
    end
end
