# Dense-only minimizer: run with jld --no-revise in an isolated OpenCL environment.
# No GPU sparse arrays, kernels, dependency setup or persistent preferences are used.
using OpenCL, SparseArrays, Random, Test

OpenCL.cl.platform!(get(ENV,"SPARSE_OPENCL_PLATFORM","NVIDIA"))
println("OPENCL_SOURCE ",pkgdir(OpenCL)," version=",pkgversion(OpenCL))
println("OPENCL_DEVICE ",OpenCL.cl.device().name,
        " driver=",OpenCL.cl.device().driver_version)
println("OPENCL_MEMORY ",OpenCL.memory_type())
OpenCL.memory_type() === OpenCL.cl.SharedVirtualMemory ||
    error("This SVM reproducer requires SharedVirtualMemory, not another memory backend")
println("JULIA_VERSION ",VERSION)
Random.seed!(0x745)

function host_fixture(T,Ti)
    dense=Matrix(dropzeros!(sprand(T,10,8,0.3)))
    dense[1,:].=rand(T,8).+one(T)
    dense[:,2].=zero(T)
    dense[3,:].=zero(T)
    host=SparseMatrixCSC{T,Ti}(sparse(dense))
    nonzeros(host)[end]=zero(T)
    return host
end

repeats=parse(Int,get(ENV,"OPENCL_TRIPLET_REPEATS","40"))
@testset "dense triplet transfer churn" begin
    for repeat in 1:repeats,T in (Float32,ComplexF32,Int32),Ti in (Int32,Int64)
        host=host_fixture(T,Ti)
        for input in (host,copy(transpose(host)))
            ptr=CLArray(SparseArrays.getcolptr(input))
            ind=CLArray(rowvals(input))
            val=CLArray(nonzeros(input))
            @test Array(val)==nonzeros(input)
            back=SparseMatrixCSC(size(input)...,Array(ptr),Array(ind),Array(val))
            @test back==input
        end
    end
end
