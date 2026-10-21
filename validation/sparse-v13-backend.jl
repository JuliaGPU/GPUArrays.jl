# Run in a task-owned backend environment through jld --no-revise.
# Arguments: backend [automatic|portable] [contracts|full] [gpuarrays-first|backend-first]
# No package setup or persistent configuration changes are performed here.
using Test, SparseArrays, LinearAlgebra

backend = get(ARGS,1,"CUDA")
policy = get(ARGS,2,"automatic")
suite = get(ARGS,3,"contracts")
order = get(ARGS,4,"gpuarrays-first")
backend in ("CUDA","AMDGPU","oneAPI","Metal","OpenCL") || error("unknown backend")
policy in ("automatic","portable") || error("unknown policy")
suite in ("contracts","full") || error("unknown suite")
order in ("gpuarrays-first","backend-first") || error("unknown import order")
load_package(name) = Core.eval(Main,Expr(:using,Expr(:.,Symbol(name))))
order=="gpuarrays-first" && load_package("GPUArrays")
if backend=="CUDA"
    load_package("CUDACore")
    # Public import scenarios, not arbitrary dependency orders: CUDACore requires
    # GPUArrays, and cuSOLVER itself requires cuSPARSE in this package stack.
    for name in (order=="gpuarrays-first" ? ("cuSPARSE","cuSOLVER") : ("cuSOLVER","cuSPARSE"))
        load_package(name)
    end
    array_type=CUDACore.CuArray
    modules=(CUDACore,cuSPARSE,cuSOLVER)
else
    load_package(backend)
    backend_module=getfield(Main,Symbol(backend))
    array_type=getfield(backend_module,Dict("AMDGPU"=>:ROCArray,"oneAPI"=>:oneArray,
                                "Metal"=>:MtlArray,"OpenCL"=>:CLArray)[backend])
    modules=(backend_module,)
end
order=="backend-first" && load_package("GPUArrays")
if backend in ("CUDA","AMDGPU","oneAPI")
    extension=Dict("CUDA"=>:CUSPARSEExt,"AMDGPU"=>:ROCSPARSEExt,"oneAPI"=>:ONEMKLExt)[backend]
    @test Base.get_extension(GPUArrays,extension)!==nothing
end
if backend=="OpenCL"
    platform=get(ENV,"SPARSE_OPENCL_PLATFORM","NVIDIA")
    if platform=="Portable Computing Language"
        load_package("pocl_jll")
    end
    OpenCL.cl.platform!(platform)
    OpenCL.program_backend!(Symbol(get(ENV,"SPARSE_OPENCL_PROGRAM_BACKEND","auto")))
    println("OPENCL_DEVICE ",OpenCL.cl.device().name)
    println("OPENCL_PROGRAM_BACKEND ",OpenCL.program_backend()," -> ",
            OpenCL.resolve_program_backend(OpenCL.cl.device()))
end
GPUArrays.allowscalar(false)
for mod in (GPUArrays,modules...)
    path=dirname(dirname(pathof(mod)))
    println("VALIDATION_PACKAGE ",nameof(mod),' ',pkgversion(mod),' ',path)
    if success(pipeline(`git -C $path rev-parse HEAD`;stdout=devnull,stderr=devnull))
        revision=readchomp(`git -C $path rev-parse HEAD`)
        dirty=!isempty(readchomp(`git -C $path status --porcelain`))
        println("VALIDATION_REVISION ",nameof(mod),' ',revision," dirty=",dirty)
    end
end
println("VALIDATION_MODE ",join((backend,policy,suite,order),' '))

const portable_selections=Ref(0)
if policy=="portable"
    # Process-local test instrumentation. Public operations must use the portable
    # executor even for vendor-supported operands; production methods are unchanged.
    @eval GPUArrays begin
        select_sparse(::Automatic,candidate) =
            ($(portable_selections)[]+=1; Portable())
        select_sparse(::Automatic,candidate::UnsupportedSparseVendor) =
            ($(portable_selections)[]+=1; Portable())
    end
end

include(joinpath(pkgdir(GPUArrays),"test","testsuite.jl"))
default_types="Float32,ComplexF32,Int32,Float16"
if backend=="OpenCL" && !("cl_khr_fp16" in OpenCL.cl.device().extensions)
    # Match the backend's device-capability contract, not a sparse-test exemption.
    # Explicit caller-selected types remain unchanged and can deliberately test errors.
    default_types="Float32,ComplexF32,Int32"
    @testset "OpenCL unsupported Float16 rejects before allocation" begin
        @test_throws ErrorException array_type{Float16}(undef,1)
    end
end
types=[getfield(Base,Symbol(name)) for name in
       split(get(ENV,"SPARSE_VALIDATION_TYPES",default_types),',')]
println("VALIDATION_FIXTURES ",suite=="full" ? string(types) :
        "fixed v13 fixtures: primarily Float32, with explicit integer/index edge cases")
@testset "$backend $policy $suite" begin
    if suite=="full"
        TestSuite.tests["sparse"](array_type;eltypes=types)
    else
        TestSuite.sparse_v13(array_type,types)
    end
    if policy=="portable" && suite=="full"
        @test portable_selections[]>0
    end
end
println("PORTABLE_SELECTIONS ",portable_selections[])

@testset "sparse extension ambiguities" begin
    ambiguities=filter(Test.detect_ambiguities(GPUArrays,SparseArrays,LinearAlgebra,modules...;
                                               recursive=true)) do methods
        any(methods) do method
            file=replace(string(method.file),'\\'=>'/')
            occursin("host/sparse/",file) || any(name->occursin(name*".jl",file) ||
                occursin("ext/"*name*"/",file),
                ("CUSPARSEExt","CUSOLVERExt","ROCSPARSEExt","ONEMKLExt"))
        end
    end
    isempty(ambiguities) || show(stdout,"text/plain",ambiguities)
    @test isempty(ambiguities)
end
