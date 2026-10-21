# Run through jld in the migrated CUDA environment and again against CUDA 6.4.2.
# SPARSE_BENCH_DATA points to cached NIST Matrix Market .mtx.gz files.
# SPARSE_BENCH_FILTER optionally restricts labels; SPARSE_BENCH_TRACE_MEMORY=1
# adds one warmed, pool-purged NVTX range per operation for Nsight memory tracing.
using CUDA,CUDACore,cuSPARSE,GPUArrays,NVTX
using SparseArrays,LinearAlgebra,Random,Statistics,SHA,Test

GPUArrays.allowscalar(false)
const modern_sparse=isdefined(GPUArrays,:Portable)
const bench_filter=Regex(get(ENV,"SPARSE_BENCH_FILTER",""))
const trace_memory=get(ENV,"SPARSE_BENCH_TRACE_MEMORY","0")=="1"
const repetitions=parse(Int,get(ENV,"SPARSE_BENCH_REPETITIONS","11"))

function read_real_symmetric(path)
    println("INPUT_SHA256\t",basename(path),'\t',bytes2hex(sha256(read(path))))
    open(`gzip -dc $path`) do io
        header=lowercase(readline(io))
        occursin("coordinate real symmetric",header) || error("unsupported Matrix Market header")
        line=readline(io)
        while startswith(line,"%") || isempty(strip(line))
            line=readline(io)
        end
        m,n,entries=parse.(Int,split(line))
        rows=Int32[];cols=Int32[];values=Float32[]
        sizehint!(rows,2entries);sizehint!(cols,2entries);sizehint!(values,2entries)
        for line in eachline(io)
            isempty(strip(line)) && continue
            i,j,v=split(line)
            row=parse(Int32,i);col=parse(Int32,j);value=parse(Float32,v)
            push!(rows,row);push!(cols,col);push!(values,value)
            if row!=col
                push!(rows,col);push!(cols,row);push!(values,value)
            end
        end
        sparse(rows,cols,values,m,n)
    end
end

function measure(label,f,reference)
    occursin(bench_filter,label) || return
    CUDACore.synchronize()
    first=NVTX.@range "first:"*label CUDACore.@timed f()
    actual=Array(first.value)
    @test actual ≈ reference rtol=2f-4 atol=1f-5
    first_seconds=first.time
    first=nothing
    for warmup in 1:3
        f()
        CUDACore.synchronize()
        GC.gc(false)
    end
    times=Float64[];host_bytes=Int[];device_bytes=Int[]
    for trial in 1:repetitions
        GC.gc(false)
        CUDACore.synchronize()
        sample=NVTX.@range "warm:"*label CUDACore.@timed f()
        push!(times,sample.time)
        push!(host_bytes,sample.cpu_bytes)
        push!(device_bytes,sample.gpu_bytes)
    end
    # gpu_bytes is cumulative requested allocation, NOT observed peak live memory.
    # It is an upper bound on incremental peak, excluding pre-existing inputs.
    println("RESULT\t",label,'\t',first_seconds,'\t',median(times),'\t',
            median(host_bytes),'\t',median(device_bytes))
    if trace_memory
        GC.gc(true)
        CUDACore.reclaim()
        CUDACore.synchronize()
        result=NVTX.@range "memory:"*label begin
            value=f()
            CUDACore.synchronize()
            value
        end
        result=nothing
    end
    nothing
end

function apply_mv!(policy,y,t,A,x)
    if modern_sparse
        GPUArrays.spmv!(y,t,A,x,one(eltype(y)),zero(eltype(y));algorithm=policy)
    else
        mul!(y,LinearAlgebra.wrap(A,t),x)
    end
end
function apply_mm!(policy,C,t,A,B)
    if modern_sparse
        GPUArrays.spmm!(C,t,'N',A,B,one(eltype(C)),zero(eltype(C));algorithm=policy)
    else
        mul!(C,LinearAlgebra.wrap(A,t),B)
    end
end
function policies()
    modern_sparse ? (("automatic",GPUArrays.Automatic()),("vendor",GPUArrays.Vendor()),
                      ("portable",GPUArrays.Portable())) : (("old-vendor",nothing),)
end

println("STACK\tJulia=",VERSION,"\tCUDA=",pkgversion(CUDA),"\tcuSPARSE=",pkgversion(cuSPARSE),
        "\tGPUArrays=",pkgversion(GPUArrays),"\truntime=",CUDA.runtime_version())
CUDA.versioninfo()
println("COLUMNS\tcase/format/operation/policy\tfirst_seconds\twarm_median_seconds\thost_alloc_bytes\tdevice_alloc_bytes")
Random.seed!(0x74513)
n=10000
cases=Pair{String,SparseMatrixCSC{Float32,Int32}}[]
push!(cases,"diagonal"=>SparseMatrixCSC{Float32,Int32}(spdiagm(0=>rand(Float32,n))))
push!(cases,"banded"=>SparseMatrixCSC{Float32,Int32}(spdiagm(
    [k=>rand(Float32,n-abs(k)) for k in -4:4]...)))
push!(cases,"uniform"=>SparseMatrixCSC{Float32,Int32}(sprand(Float32,n,n,8/n)))
rows=clamp.(floor.(Int,exp.(rand(80000).*log(n))),1,n)
push!(cases,"skewed"=>SparseMatrixCSC{Float32,Int32}(sparse(rows,rand(1:n,80000),rand(Float32,80000),n,n)))
push!(cases,"empty-rows"=>SparseMatrixCSC{Float32,Int32}(sparse(4 .* rand(1:div(n,4),80000),
    rand(1:n,80000),rand(Float32,80000),n,n)))
data=get(ENV,"SPARSE_BENCH_DATA",joinpath(homedir(),"Julia/tmp/ga13-benchmark-data"))
for name in ("1138_bus","bcsstk17")
    path=joinpath(data,name*".mtx.gz")
    isfile(path) || error("Cache $path from math.nist.gov before running")
    push!(cases,name=>SparseMatrixCSC{Float32,Int32}(read_real_symmetric(path)))
end

for (name,host) in cases, (format,S) in (("csr",CuSparseMatrixCSR),("csc",CuSparseMatrixCSC))
    println("CASE\t",name,'\t',format,'\t',size(host),'\t',nnz(host))
    A=S(host)
    for trans in ('N','T')
        H=LinearAlgebra.wrap(host,trans)
        x=CuArray(rand(Float32,size(H,2)));y=CuArray{Float32}(undef,size(H,1))
        reference=H*Array(x)
        for (policy_name,policy) in policies()
            label="$name/$format/spmv-$trans/$policy_name"
            measure(label,()->apply_mv!(policy,y,trans,A,x),reference)
        end
        for width in (1,8,64)
            B=CuArray(rand(Float32,size(H,2),width));C=CuArray{Float32}(undef,size(H,1),width)
            reference=H*Array(B)
            for (policy_name,policy) in policies()
                label="$name/$format/spmm-$trans-w$width/$policy_name"
                measure(label,()->apply_mm!(policy,C,trans,A,B),reference)
            end
        end
    end
end

host=ComplexF32.(last(cases[2]),0.25f0 .* last(cases[2]))
for (format,S) in (("csr",CuSparseMatrixCSR),("csc",CuSparseMatrixCSC))
    A=S(host);B=CuArray(rand(ComplexF32,size(host,1),8));C=similar(B)
    reference=adjoint(host)*Array(B)
    for (policy_name,policy) in policies()
        measure("complex-banded/$format/spmm-C-w8/$policy_name",
                ()->apply_mm!(policy,C,'C',A,B),reference)
    end
end

# Duplicate-heavy assembly uses the same raw coordinate/value inputs in both stacks.
I=Int32.(rand(1:1000,10000));J=Int32.(rand(1:1000,10000));V=rand(Float32,10000)
I=repeat(I,8);J=repeat(J,8);V=repeat(V,8)
dI=CuArray(I);dJ=CuArray(J);dV=CuArray(V)
reference=Matrix(sparse(I,J,V,1000,1000))
measure("duplicates/assembly/csr",()->sparse(dI,dJ,dV,1000,1000;fmt=:csr),reference)

# 8,388,608 candidate products but only 262,144 final entries.
left=sparse(repeat(1:512,inner=32),repeat(1:32,512),ones(Float32,512*32),512,512)
right=sparse(repeat(1:32,inner=512),repeat(1:512,32),ones(Float32,512*32),512,512)
A=CuSparseMatrixCSR(left);B=CuSparseMatrixCSR(right)
reference=Matrix(left*right)
for (policy_name,policy) in policies()
    product=modern_sparse ? ()->GPUArrays.spgemm(A,B;algorithm=policy) : ()->A*B
    measure("candidate-expansion/csr/spgemm/$policy_name",product,reference)
end
if modern_sparse
    @test_throws ArgumentError GPUArrays.spgemm(A,B;algorithm=GPUArrays.Portable(),max_temp_bytes=1<<20)
    println("RESOURCE_LIMIT\tcandidate-expansion\t1048576\trejected")
end
