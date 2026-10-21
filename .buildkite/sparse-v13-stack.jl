# TEMPORARY: validate the unreleased v13 stack, not registered release resolution.
# Explicit path development is intentional: Julia 1.10 ignores [sources].
using Pkg, TOML

# Preferences are keyed by package name but Base needs a UUID/name binding in
# each active project, including for transitive JLLs such as CUDA_Compiler_jll.
# Extras provide that binding without changing the resolved dependency graph.
function stack_bind_preferences()
    project=Base.active_project()
    preferences=joinpath(dirname(project),"LocalPreferences.toml")
    isfile(preferences) || return
    names=Dict(info.name=>string(uuid) for (uuid,info) in Pkg.dependencies())
    data=TOML.parsefile(project)
    changed=false
    for name in keys(TOML.parsefile(preferences))
        haskey(names,name) || continue
        (get(data,"name",nothing)==name || haskey(get(data,"deps",Dict()),name)) && continue
        extras=get!(data,"extras",Dict{String,Any}())
        if !haskey(extras,name)
            extras[name]=names[name]
            changed=true
            println("STACK_PREFERENCE_BINDING ",name,' ',names[name])
        else
            extras[name]==names[name] || error("conflicting preference UUID for $name")
        end
    end
    if changed
        open(project,"w") do io
            TOML.print(io,data)
        end
    end
end

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
    source=realpath(ENV["GPUARRAYS_V13_STACK_BACKEND"])
    TOML.parsefile(joinpath(source,"Project.toml"))["name"]==backend || error("wrong backend worktree")
    isempty(readchomp(`git -C $source diff HEAD --`)) || error("commit backend changes before stack validation")
    revision=readchomp(`git -C $source rev-parse HEAD`)
    # Ignore stale workspace manifests and generated files in a reused CI checkout.
    # Test the recorded commit, without modifying the supplied working tree.
    path=joinpath(scratch,backend)
    run(`git clone --quiet --shared --no-checkout $source $path`)
    run(`git -C $path checkout --quiet --detach $revision`)
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
    package==gpuarrays || cp(ENV["GPUARRAYS_V13_STACK_PREFERENCES"],
                             joinpath(package,"LocalPreferences.toml");force=true)
end
const stack_oneapi_debug=backend=="oneAPI" &&
    get(ENV,"GPUARRAYS_V13_STACK_ONEAPI_DEBUG","false")=="true"
stack_bind_preferences()

if backend == "oneAPI" && (!setup_only || get(ENV,"GPUARRAYS_V13_STACK_BUILD_SUPPORT","false")=="true")
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
            # This job requires the changed wrappers, regardless of JLL tag timestamps.
            Base.include(build_module,joinpath(package,"deps","build_local.jl"))
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
function stack_prepare_tests()
    # CUDA's runner switches into each library test project. Julia 1.10 ignores
    # [sources] and workspaces, so developing the libraries in `active` alone does
    # not resolve test-only imports such as SparseMatricesCSR. Resolve every actual
    # test project using the same pins, including the backend's root test project.
    test_projects=String[]
    package==gpuarrays || push!(test_projects,joinpath(package,"test"))
    if backend=="CUDA"
        append!(test_projects,[joinpath(package,"lib",name,"test") for name in
            ("cublas","cusparse","cusolver","cufft","curand","cudnn",
             "cutensor","cutensornet","custatevec")])
    end
    for project in test_projects
        isfile(joinpath(project,"Project.toml")) || continue
        Pkg.activate(project)
        withenv("JULIA_PKG_PRECOMPILE_AUTO"=>"0") do
            Pkg.develop(specs)
        end
        preferences=joinpath(active,"LocalPreferences.toml")
        isfile(preferences) && cp(preferences,joinpath(project,"LocalPreferences.toml");force=true)
        stack_bind_preferences()
        println("STACK_TEST_PROJECT ",project)
    end
    Pkg.activate(active)
end

# Custom CI jobs (minimum dependencies, Enzyme) can prepare the pinned stack,
# change dependencies in its disposable checkout, then run the same checked runner.
function stack_load_backend(test_args)
    stack_bind_preferences()
    if backend=="CUDA"
        @eval using CUDA,cuSPARSE,cuSOLVER
    elseif backend=="AMDGPU"
        @eval using AMDGPU
    elseif backend=="Metal"
        @eval using Metal
    elseif backend=="oneAPI"
        @eval using oneAPI
    elseif backend=="OpenCL"
        requested=findfirst(arg->startswith(arg,"--platform="),test_args)
        request=requested===nothing ? "pocl" : split(test_args[requested],'=';limit=2)[2]
        driver=request=="pocl_next" ? :pocl_next_jll :
               request=="intel_cpu" ? :intel_opencl_rt_jll : :pocl_jll
        Pkg.add(string(driver))
        @eval using OpenCL
        @eval import $driver
    end
end
function stack_run_tests(test_args)
    stack_load_backend(test_args)
    # Imports above can install new extension methods; enter their world explicitly.
    Base.invokelatest(stack_run_loaded_tests,test_args)
end
function stack_run_loaded_tests(test_args)
    # A backend runner may otherwise skip unavailable hardware/libraries and exit 0.
    if backend=="CUDA"
        CUDA.functional() && cuSPARSE.functional() && cuSOLVER.functional() ||
            error("CUDA/cuSPARSE/cuSOLVER must be functional for stack validation")
        memory=get(Base.get_preferences(Base.PkgId(CUDA.CUDACore).uuid),"default_memory","device")
        expected=Dict("device"=>CUDA.DeviceMemory,"unified"=>CUDA.UnifiedMemory,"host"=>CUDA.HostMemory)[memory]
        CUDA.CUDACore.default_memory===expected || error("CUDA default memory preference was not propagated")
        println("STACK_CUDA_MEMORY ",memory,' ',expected)
    elseif backend=="AMDGPU"
        all(AMDGPU.functional,(:hip,:rocsparse,:rocblas,:rocsolver)) ||
            error("HIP/rocSPARSE/rocBLAS/rocSOLVER must be functional for stack validation")
    elseif backend=="Metal"
        Metal.functional() || error("Metal must be functional for stack validation")
    elseif backend=="oneAPI"
        oneAPI.functional() || error("oneAPI must be functional for stack validation")
        oneAPI.Support._check_sparse_abi()
        if stack_oneapi_debug
            for library in (oneAPI.oneL0.NEO_jll,oneAPI.oneL0.NEO_jll.libigc_jll)
                get(Base.get_preferences(Base.PkgId(library).uuid),"debug",nothing)=="true" ||
                    error("oneAPI debug preference was not propagated for $(nameof(library))")
                println("STACK_DEBUG_PREFERENCE ",nameof(library)," true")
            end
        end
    elseif backend=="OpenCL"
        requested=findfirst(arg->startswith(arg,"--platform="),test_args)
        request=requested===nothing ? "pocl" : split(test_args[requested],'=';limit=2)[2]
        any(OpenCL.cl.platforms()) do candidate_platform
            name=lowercase(candidate_platform.name*" "*candidate_platform.vendor)
            matches=request in ("pocl","pocl_next") ? occursin("portable",name) :
                    request=="intel_cpu" ? candidate_platform.name=="Intel(R) OpenCL" : occursin(lowercase(request),name)
            matches && !isempty(OpenCL.cl.devices(candidate_platform))
        end || error("requested OpenCL platform must expose a device for stack validation")
    end
    coverage=haskey(ENV,"GPUARRAYS_V13_STACK_BACKEND") ? "user" : "@"*gpuarrays
    julia_args=Cmd(Base.shell_split(get(ENV,"GPUARRAYS_V13_STACK_JULIA_ARGS","")))
    allow_reresolve=parse(Bool,get(ENV,"GPUARRAYS_V13_STACK_ALLOW_RERESOLVE","true"))
    try
        if VERSION<v"1.12" && package!=gpuarrays
            # Run after custom dependency changes, which can recreate this manifest.
            # Older Pkg rejects repeated fixed UUIDs even at identical paths. Keep
            # each generated root-test manifest as evidence; library test manifests
            # remain for runner switches. Pkg.test builds its sandbox from `active`.
            manifest=joinpath(package,"test","Manifest.toml")
            if isfile(manifest)
                evidence=mktempdir(scratch;prefix="root-test-manifest-")
                mv(manifest,joinpath(evidence,"Manifest.toml"))
            end
        end
        Pkg.test([backend == "JLArrays" ? "GPUArrays" : backend];
                 coverage,test_args,julia_args,allow_reresolve)
    finally
        if haskey(ENV,"GPUARRAYS_V13_STACK_BACKEND") && package!=gpuarrays
            # Coverage is emitted next to the disposable clone's sources. Preserve it
            # for the backend's normal post-command coverage uploader before cleanup.
            destination=realpath(ENV["GPUARRAYS_V13_STACK_BACKEND"])
            for (root,dirs,files) in walkdir(package), file in files
                endswith(file,".cov") || continue
                target=joinpath(destination,relpath(joinpath(root,file),package))
                mkpath(dirname(target))
                cp(joinpath(root,file),target;force=true)
            end
        end
    end
end

if !setup_only || get(ENV,"GPUARRAYS_V13_STACK_PREPARE_TESTS","false")=="true"
    stack_prepare_tests()
end
if !setup_only
    # CUDA defaults omit library suites; Metal defaults omit GPUArrays' suite.
    test_args = backend in ("CUDA","Metal") ? ["--all","--verbose"] :
                backend == "OpenCL" ? ["--platform=pocl","--verbose"] : ["--verbose"]
    length(ARGS)>1 && (test_args=ARGS[2:end])
    stack_run_tests(test_args)
end
