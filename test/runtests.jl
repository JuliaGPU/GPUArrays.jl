using GPUArrays, Test
using GPUArrays.TestSuite

using Pkg

@testset "JLArray" begin
    GPUArrays.test(JLArray)
end

function test_package(package)
    mktempdir() do devdir
        withenv("JULIA_PKG_DEVDIR" => devdir) do
            # try to install from the same branch of GPUArrays
            try
                branch = chomp(read(`git -C $(@__DIR__) rev-parse --abbrev-ref HEAD`, String))
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
    @testset "CuArray" begin
        test_package("CuArrays")
    end
end
