# TEMPORARY: validate the unreleased v13 stack, not registered release resolution.
# Explicit path development is intentional: Julia 1.10 ignores [sources].
using Pkg, TOML

gpuarrays = normpath(joinpath(@__DIR__, ".."))
pins = TOML.parsefile(joinpath(@__DIR__, "sparse-v13-stack.toml"))
backend = isempty(ARGS) ? "JLArrays" : first(ARGS)
setup_only = "--setup-only" in ARGS
scratch = mktempdir(get(ENV,"GPUARRAYS_V13_STACK_PARENT",tempdir());
                     prefix="gpuarrays-v13-stack-",cleanup=!setup_only)

function checkout(name)
    pin = pins[name]
    path = joinpath(scratch, name)
    run(`git clone --quiet --filter=blob:none $(pin["url"]) $path`)
    run(`git -C $path checkout --quiet --detach $(pin["rev"])`)
    revision = readchomp(`git -C $path rev-parse HEAD`)
    revision == pin["rev"] || error("Unexpected $name revision: $revision")
    println("STACK_REVISION $name $revision")
    path
end

println("STACK_REVISION GPUArrays ", readchomp(`git -C $gpuarrays rev-parse HEAD`))
println("STACK_DIRTY GPUArrays ", !isempty(readchomp(`git -C $gpuarrays status --porcelain`)))
println("STACK_JULIA ", VERSION)
specs = PackageSpec[
    PackageSpec(path=gpuarrays),
    PackageSpec(path=joinpath(gpuarrays,"lib","GPUArraysCore")),
    PackageSpec(path=checkout("AcceleratedKernels")),
    PackageSpec(path=checkout("Atomix")),
]
backend in ("JLArrays","docs") && push!(specs,PackageSpec(path=joinpath(gpuarrays,"lib","JLArrays")))
package = if backend in ("JLArrays","docs")
    gpuarrays
elseif haskey(ENV,"GPUARRAYS_V13_STACK_BACKEND")
    path=realpath(ENV["GPUARRAYS_V13_STACK_BACKEND"])
    TOML.parsefile(joinpath(path,"Project.toml"))["name"]==backend || error("wrong backend worktree")
    println("STACK_REVISION $backend ",readchomp(`git -C $path rev-parse HEAD`))
    path
else
    checkout(backend)
end
if package != gpuarrays
    push!(specs,PackageSpec(path=package))
end
if backend == "CUDA"
    # Include test-only workspace packages too. Otherwise 1.10 can silently resolve
    # a registered library instead of the checked-out cuSPARSE/cuSOLVER stack.
    for relative in ("CUDACore","CUDATools","lib/cupti","lib/nvml",
                     "lib/cublas","lib/cusparse","lib/cusolver","lib/cufft","lib/curand",
                     "lib/cudnn","lib/cutensor","lib/cutensornet","lib/custatevec")
        push!(specs,PackageSpec(path=joinpath(package,relative)))
    end
elseif backend == "OpenCL"
    push!(specs,PackageSpec(path=joinpath(package,"lib","intrinsics")))
end

active = get(ENV,"GPUARRAYS_V13_STACK_SETUP_PROJECT",
             backend == "docs" ? joinpath(gpuarrays,"docs") : joinpath(scratch,"env"))
filter!(spec -> normpath(spec.path) != active,specs)
Pkg.activate(active)
withenv("JULIA_PKG_PRECOMPILE_AUTO"=>"0") do
    Pkg.develop(specs)
end
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
if haskey(ENV,"GPUARRAYS_V13_STACK_PREFERENCES")
    cp(ENV["GPUARRAYS_V13_STACK_PREFERENCES"],joinpath(active,"LocalPreferences.toml");force=true)
end

if !setup_only && backend == "oneAPI"
    # This branch changes the support C++ wrappers. Build them for stack validation;
    # the registered JLL has not incorporated those changes yet.
    Pkg.activate(package)
    Pkg.develop(filter(spec -> normpath(spec.path) != package,specs))
    Pkg.activate(joinpath(package,"deps"))
    Pkg.instantiate()
    @eval using Scratch
    original_args=copy(ARGS)
    empty!(ARGS)
    push!(ARGS,joinpath(scratch,"support-build"))
    try
        Scratch.with_scratch_directory(joinpath(scratch,"support-scratch")) do
            build_module=Module(gensym(:SparseV13SupportBuild))
            Core.eval(build_module,:(include(path)=Base.include($build_module,path)))
            Base.include(build_module,joinpath(package,"deps","build_ci.jl"))
        end
    finally
        empty!(ARGS)
        append!(ARGS,original_args)
    end
    Pkg.activate(active)
    Pkg.add("oneAPI_Support_jll")
    preferences=joinpath(package,"LocalPreferences.toml")
    cp(preferences,joinpath(active,"LocalPreferences.toml");force=true)
    support=TOML.parsefile(preferences)["oneAPI_Support_jll"]["liboneapi_support_path"]
    ENV["GPUARRAYS_V13_STACK_SUPPORT"]=support
    println("STACK_SUPPORT_LIBRARY ",support)
end
if !setup_only
    # A backend runner may otherwise skip unavailable hardware/libraries and exit 0.
    if backend=="CUDA"
        @eval using CUDA,cuSPARSE,cuSOLVER
        CUDA.functional() && cuSPARSE.functional() && cuSOLVER.functional() ||
            error("CUDA/cuSPARSE/cuSOLVER must be functional for stack validation")
    elseif backend=="AMDGPU"
        @eval using AMDGPU
        all(AMDGPU.functional,(:hip,:rocsparse,:rocblas,:rocsolver)) ||
            error("HIP/rocSPARSE/rocBLAS/rocSOLVER must be functional for stack validation")
    elseif backend=="Metal"
        @eval using Metal
        Metal.functional() || error("Metal must be functional for stack validation")
    elseif backend=="oneAPI"
        @eval using oneAPI
        oneAPI.functional() || error("oneAPI must be functional for stack validation")
        oneAPI.Support._check_sparse_abi()
    elseif backend=="OpenCL"
        Pkg.add("pocl_jll")
        @eval using OpenCL,pocl_jll
        any(OpenCL.cl.platforms()) do platform
            occursin("portable",lowercase(platform.name)) && !isempty(OpenCL.cl.devices(platform))
        end || error("PoCL must expose a device for stack validation")
    end
    # CUDA defaults omit library suites; Metal defaults omit GPUArrays' suite.
    test_args = backend in ("CUDA","Metal") ? ["--all","--verbose"] :
                backend == "OpenCL" ? ["pocl/"] : ["--verbose"]
    length(ARGS)>1 && (test_args=ARGS[2:end])
    coverage=haskey(ENV,"GPUARRAYS_V13_STACK_BACKEND") ? "user" : "@"*gpuarrays
    Pkg.test([backend == "JLArrays" ? "GPUArrays" : backend]; coverage,test_args)
end
