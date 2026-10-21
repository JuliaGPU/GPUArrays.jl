using Test, GPUArrays, JLArrays, SparseArrays, LinearAlgebra, Adapt
using GPUArrays: GPUSparseMatrixCSR, GPUSparseMatrixCSC, GPUSparseMatrixCOO, GPUSparseVector

GPUArrays.allowscalar(false)

struct FixedSparseStorage{T} <: GPUArrays.AbstractGPUArray{T,1}
    data::JLVector{T}
end
Base.size(x::FixedSparseStorage) = size(x.data)
Base.Array(x::FixedSparseStorage) = Array(x.data)
GPUArrays.KernelAbstractions.get_backend(x::FixedSparseStorage) = GPUArrays.get_backend(x.data)
Base.resize!(::FixedSparseStorage, ::Integer) = throw(ArgumentError("fixed storage"))

# Same CPU-transfer policy as MLDataDevices' GPUArraysSparseArraysExt, without
# resolving a downstream release whose compatibility does not yet include v13.
function downstream_cpu_transfer(::Type{T},x::Union{GPUSparseMatrixCSC,GPUSparseVector}) where T
    target=x isa GPUSparseMatrixCSC ? SparseMatrixCSC : SparseVector
    if T <: AbstractFloat
        eltype(x) <: Complex && return target{Complex{T}}(x)
        return target{T}(x)
    end
    return target(x)
end

@testset "v13 sparse regressions" begin
    @testset "host-accessible mixed sparse broadcast rejects" begin
        x=GPUSparseVector(Int32[1,2],Float32[3,4],2)
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO)
            A=S===GPUSparseMatrixCOO ? S(Int32[1,2],Int32[1,2],Float32[1,2],(2,2)) :
                                      S(Int32[1,2,3],Int32[1,2],Float32[1,2],(2,2))
            before=Array(A)
            @test_throws ErrorException A .* x
            @test_throws ErrorException x .+ A
            @test_throws ErrorException 2 .* A .+ x
            @test_throws ErrorException (A .= A .* x)
            @test Array(A)==before
        end
    end
    @testset "downstream dispatch and storage migration" begin
        for (name,S) in ((:AbstractGPUSparseMatrixCSR,GPUSparseMatrixCSR),
                         (:AbstractGPUSparseMatrixCSC,GPUSparseMatrixCSC),
                         (:AbstractGPUSparseMatrixCOO,GPUSparseMatrixCOO))
            legacy=getfield(GPUArrays,name)
            @test legacy===S
            @test legacy{Float32,Int32}===S{Float32,Int32}
            @test !isabstracttype(legacy)
        end
        for host in (sparse(Float32[1 0;0 2]),sparse(ComplexF32[1 im;0 2]),
                     sparsevec([1,3],Float32[1,2],4),sparsevec([1,3],ComplexF32[1,im],4))
            A=adapt(JLArray,host)
            for T in (Float32,Float64,Nothing)
                B=downstream_cpu_transfer(T,A)
                expected=T <: AbstractFloat ? (eltype(host) <: Complex ? Complex{T} : T) : eltype(host)
                @test B isa typeof(host).name.wrapper
                @test eltype(B)===expected
                @test B==host
            end
            dense=similar(nonzeros(A),eltype(A),size(A))
            fill!(dense,convert(eltype(A),NaN))
            copyto!(dense,A)
            @test dense isa JLArray
            @test Array(dense)==Array(host)
        end
    end
    @testset "generalized dot dimension preflight" begin
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO)
            A=S(adapt(JLArray,sparse(Int32[1 0 2;3 4 0])))
            x=JLArray(Int32[1,2]);y=JLArray(Int32[1,2,3])
            @test dot(x,A,y)==Int32(29)
            @test_throws DimensionMismatch dot(JLArray(Int32[1,2,3]),A,y)
            @test_throws DimensionMismatch dot(JLArray(Int32[1]),A,y)
            @test_throws DimensionMismatch dot(x,A,JLArray(Int32[1,2]))
            @test_throws DimensionMismatch dot(x,A,JLArray(Int32[1,2,3,4]))
            empty=S(adapt(JLArray,spzeros(Int32,0,0)))
            @test_throws DimensionMismatch dot(JLArray(Int32[1]),empty,JLArray(Int32[]))
            @test_throws DimensionMismatch dot(JLArray(Int32[]),empty,JLArray(Int32[1]))
        end
        @test_throws DimensionMismatch dot(JLArray(Int32[1,2]),JLArray(Int32[3]))
    end
    @testset "diagonal wrapper and offset compatibility" begin
        host=sparse(ComplexF32[1 im 0;2 3 4])
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO)
            A=S(adapt(JLArray,host))
            for wrap in (identity,transpose,adjoint), k in (-20,-2,-1,0,1,3,20,typemin(Int),typemax(Int))
                if k in (typemin(Int),typemax(Int))
                    @test isempty(diag(wrap(A),k))
                else
                    @test Array(diag(wrap(A),k))==diag(Matrix(wrap(host)),k)
                end
            end
            @test Array(diag(A,UInt(1)))==diag(Matrix(host),1)
            @test Array(diag(transpose(A),Int8(-1)))==diag(Matrix(transpose(host)),-1)
            @test_throws InexactError diag(A,typemax(UInt))
        end
    end
    @testset "preflight narrowing before compression" begin
        coo=GPUSparseMatrixCOO(JLArray(Int64[]),JLArray(Int64[]),JLArray(Float32[]),(2^32,1))
        @test_throws ArgumentError GPUSparseMatrixCSR{Float32,Int32}(coo)
        @test_throws ArgumentError GPUSparseMatrixCSC{Float32,Int32}(coo)
        wide=GPUSparseMatrixCOO(JLArray(Int64[]),JLArray(Int64[]),JLArray(Float32[]),(1,2^32))
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC), wrap in (transpose,adjoint)
            @test_throws ArgumentError S{Float32,Int32}(wrap(wide))
        end
        bad=GPUSparseMatrixCOO(JLArray(Int64[1]),JLArray(Int64[1]),JLArray(ComplexF32[1+im]),(2^32,1))
        @test_throws InexactError GPUSparseMatrixCSR{Float32}(bad)
        @test_throws InexactError GPUSparseMatrixCSC{Float32}(transpose(bad))
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC)
            @test_throws ArgumentError S{Float32,Int8}(Diagonal(JLArray(ones(Float32,127))))
            @test_throws ArgumentError S{Float32,Int8}(Diagonal(JLArray(zeros(Float32,127))))
            D=S{Float32,Int8}(Diagonal(JLArray(zeros(Float32,126))))
            @test nnz(D)==126 && size(D)==(126,126) && all(iszero,nonzeros(D))
        end
        tiny=GPUSparseMatrixCOO(JLArray(Int8[1,1]),JLArray(Int8[1,2]),JLArray(Float32[1,2]),(1,2))
        @test_throws ArgumentError GPUSparseMatrixCSR{Float32,Bool}(tiny)
    end
    @testset "rectangular scaling and promoted wrapped products" begin
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO)
            host_cancel=sparse(reshape(Float32[16777216,1,-16777216],1,3))
            cancel=S(adapt(JLArray,host_cancel))
            rhs=JLArray(ones(Float32,3,1))
            @test eltype(1.0*cancel*rhs)===Float64
            @test Array(1.0*cancel*rhs)==ones(1,1)
            @test Array(1.0*transpose(copy(transpose(cancel)))*rhs)==ones(1,1)
            for dims in ((2,3),(3,2),(0,3),(3,0))
                host=sparse(reshape(Float32.(1:prod(dims)),dims))
                A=S(adapt(JLArray,host))
                # Preserve the existing GPU rectangular diagonal operation even
                # when newer SparseArrays versions restrict UniformScaling to square.
                for J in (I,2.5f0*I,-3f0*I), op in (+,-)
                    identity=zeros(Float32,dims)
                    for i in 1:min(dims...)
                        identity[i,i]=J.λ
                    end
                    @test Array(op(A,J)) == op(Matrix(host),identity)
                    @test Array(op(J,A)) == op(identity,Matrix(host))
                end
                @test_throws DimensionMismatch one(A)
                @test_throws DimensionMismatch oneunit(A)
            end
            for dims in ((0,0),(2,2))
                host=sparse(reshape(Float32.(1:prod(dims)),dims))
                A=S(adapt(JLArray,host))
                for J in (I,2.5f0*I,-3f0*I), op in (+,-)
                    @test Array(op(A,J)) == Matrix(op(host,J))
                    @test Array(op(J,A)) == Matrix(op(J,host))
                end
            end
            host=sparse(Float32[1.1 2.3;0 3.7])
            A=S(adapt(JLArray,host))
            for wrap in (Symmetric,Hermitian)
                @test Array(2*wrap(A))==Matrix(2*wrap(host))
                @test Array(wrap(A)*2)==Matrix(wrap(host)*2)
                @test sparse(wrap(A)) isa S
                @test Array(sparse(wrap(A)))==Matrix(wrap(host))
                @test SparseMatrixCSC(wrap(A))==sparse(wrap(host))
                for op in (+,-), outer in (identity,transpose,adjoint)
                    @test Array(op(outer(A),wrap(A)))≈Matrix(op(outer(host),wrap(host)))
                    @test Array(op(wrap(A),outer(A)))≈Matrix(op(wrap(host),outer(host)))
                end
            end
            for wrap in (transpose,adjoint)
                @test Array(1.1*wrap(A)) ≈ Matrix(1.1*wrap(host))
                @test Array(wrap(A)*1.1) ≈ Matrix(wrap(host)*1.1)
                @test Array(1.1*wrap(A)*A) ≈ Matrix((1.1*wrap(host))*host)
                @test Array(wrap(A)^2) ≈ Matrix(wrap(host)^2)
            end
        end
    end
    @testset "backend arithmetic parity" begin
        host=sparse(ComplexF32[2 0;1+im 3])
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO)
            A=S(adapt(JLArray,host))
            for B in (A,transpose(A),adjoint(A)), op in (+,-)
                reference=B === A ? host : B isa Transpose ? transpose(host) : adjoint(host)
                @test Array(op(A,B)) ≈ Array(op(host,reference))
                @test Array(op(B,A)) ≈ Array(op(reference,host))
                @test Array(op(B,2I)) ≈ Array(op(reference,2I))
                @test Array(op(2I,B)) ≈ Array(op(2I,reference))
                D=Diagonal(ComplexF32[2,4])
                @test Array(op(B,D)) ≈ Array(op(reference,D))
                @test Array(op(D,B)) ≈ Array(op(D,reference))
            end
            @test Array(one(A))==Matrix{ComplexF32}(I,2,2)
            @test Array(oneunit(A))==Matrix{ComplexF32}(I,2,2)
            @test Array(A*2I)==Matrix(host)*2I
            @test Array(2I*A)==2I*Matrix(host)
            D=Diagonal(ComplexF32[2,4])
            @test Array(A*D)==Matrix(host*D)
            @test Array(D*A)==Matrix(D*host)
            @test Array(kron(A,D))≈Matrix(kron(host,D))
            @test Array(kron(D,A))≈Matrix(kron(D,host))
            @test iszero(zero(A))
        end
        @test SparseMatrixCSC(JLArray(Matrix(host)))==host
        host32=sparse(Int32[typemax(Int32) 0;0 typemax(Int32)])
        @test tr(adapt(JLArray,host32))===tr(host32)
        @test Array(sparse(JLArray(Int32[1]),JLArray(Int32[1]),JLArray(Float32[2]),Int32(1),Int32(1)))==reshape(Float32[2],1,1)
    end
    @testset "borrowed CSC column accessors" begin
        A=GPUSparseMatrixCSC(JLArray(Int64[1,3,3,4]),JLArray(Int32[1,3,2]),
                             JLArray(Float32[2,4,6]),(3,3))
        for j in 1:3
            column=view(A,:,j)
            GPUArrays.@allowscalar begin
                @test nnz(column)==(j==1 ? 2 : j==2 ? 0 : 1)
                @test Array(nonzeros(column))==Array(nonzeros(A))[nzrange(A,j)]
                @test Array(rowvals(column))==Array(rowvals(A))[nzrange(A,j)]
            end
        end
    end
    @testset "maximum coordinate" begin
        x = GPUSparseVector(JLArray(Int8[127]), JLArray(Float32[2]), 127)
        y = x .+ x
        @test nnz(y) == 1
        @test Array(y)[end] == 4
        @test Array(GPUArrays.compress(JLArray(Int8[127]), 127))[end] == 2
    end
    @testset "rejected mutation preserves every buffer" begin
        p = JLArray([1, 2, 3]); i = JLArray([1, 2])
        v = JLArray([1., 7., 0., 9.])
        A = GPUSparseMatrixCSC(p, i, view(v, 1:2:4), (2, 2))
        before = (Array(p), Array(i), Array(v))
        @test_throws ArgumentError dropzeros!(A)
        @test (Array(p), Array(i), Array(v)) == before
        p = JLArray([1, 2, 3]); i = JLArray([1, 2])
        v = FixedSparseStorage(JLArray(Float32[1, 0]))
        dest = GPUSparseMatrixCSC(p, i, v, (2, 2))
        src = GPUSparseMatrixCSC(JLArray([1, 2, 2]), JLArray([1]), JLArray(Float32[1]), (2, 2))
        before = (Array(p), Array(i), Array(v))
        @test_throws ArgumentError copyto!(dest, src)
        @test (Array(p), Array(i), Array(v)) == before
    end
    @testset "symmetric nonfinite filtering" begin
        S = sparse([1. 2.; 4. 3.])
        A = adapt(JLArray, S)
        for wrapper in (Symmetric, Hermitian), uplo in (:U, :L)
            @test Array(wrapper(A, uplo) * JLArray([Inf, 1.])) == wrapper(S, uplo) * [Inf, 1.]
        end
    end
    @testset "scaled norms" begin
        for v in (1f30, 1f-30)
            S = sparse([1], [1], [v], 1, 1)
            @test norm(adapt(JLArray, S)) ≈ norm(S)
        end
    end
    @testset "component aliasing" begin
        A = adapt(JLArray, sparse(Float32[1 0; 2 3]))
        B = GPUSparseMatrixCSC(A.colPtr, A.rowVal, copy(A.nzVal), size(A))
        @test Base.mightalias(A, B)
        @test !Base.mightalias(A, copy(A))
    end
    @testset "independent compressed widths" begin
        for S in (GPUSparseMatrixCSR, GPUSparseMatrixCSC)
            A = S(JLArray(Int64[1, 3, 4]), JLArray(Int32[1, 2, 2]),
                  JLArray(Float32[1, 2, 3]), (2, 2))
            other = S === GPUSparseMatrixCSR ? GPUSparseMatrixCSC : GPUSparseMatrixCSR
            for B in (copy(A), similar(A), similar(A, Float64), S{Float64}(A), other(A))
                @test eltype(GPUArrays.compressed_ptr(B)) == Int64
                @test SparseArrays.indtype(B) == Int32
                @test !Base.mightalias(A, B)
            end
            for B in (S{Float64,Int32}(A), similar(A, Float64, Int32))
                @test eltype(GPUArrays.compressed_ptr(B)) == Int32
                @test SparseArrays.indtype(B) == Int32
            end
            @test SparseMatrixCSC(A) == SparseMatrixCSC(adapt(Array, A))
            @test eltype(SparseArrays.getcolptr(SparseMatrixCSC(A))) == Int64
            @test Array(GPUArrays.check_structure(A)) == Array(A)
            @test Array(other(A)) == Array(A)
        end
        # Neither vector nor COO needs a terminal pointer in its coordinate type.
        x = GPUSparseVector(JLArray(Int8.(1:127)), JLArray(ones(Float32, 127)), 127)
        @test nnz(GPUArrays.check_structure(x)) == 127
        I = repeat(Int8.(1:20); inner=20)
        J = repeat(Int8.(1:20); outer=20)
        A = GPUSparseMatrixCOO(JLArray(I), JLArray(J), JLArray(ones(Float32, 400)), (20, 20))
        @test nnz(A) == 400
        B = GPUSparseMatrixCSR(A)
        @test eltype(B.rowPtr) == Int
        @test SparseArrays.indtype(B) == Int8
        @test nnz(GPUArrays.check_structure(B)) == 400
        @test_throws ArgumentError GPUSparseMatrixCSR{Float32,Int8}(B)
        @test eltype(similar(x, Float32, (10, 10)).colPtr) == Int
        @test eltype(similar(x, Float32, Int8, (10, 10)).colPtr) == Int8
        # The pointer-buffer length does not have to fit its element type.
        ptr = ones(Int8, 201); ptr[end] = 2
        A = GPUSparseMatrixCSR(JLArray(ptr), JLArray(Int16[1]), JLArray(Float32[2]), (200, 1))
        B = GPUSparseMatrixCSR(JLArray(Int8[1, 2]), JLArray(Int16[1]), JLArray(Float32[3]), (1, 1))
        C = A * B
        @test eltype(C.rowPtr) == Int8
        @test SparseArrays.indtype(C) == Int16
        @test Array(C)[end] == 6
    end
    @testset "scalar equality can exceed input pointer widths" begin
        a = sparse(ones(Int8, 72), Int8.(1:72), zeros(Float32, 72), 2, 72)
        b = sparse(fill(Int8(2), 72), Int8.(1:72), zeros(Float32, 72), 2, 72)
        A = GPUSparseMatrixCSR{Float32,Int8}(adapt(JLArray, a))
        B = GPUSparseMatrixCSR{Float32,Int8}(adapt(JLArray, b))
        @test A == B
        @test_throws ArgumentError A .+ B
    end
    @testset "integer value conversion is checked" begin
        x = GPUSparseVector(JLArray(Int32[1]), JLArray(Int64[300]), 2)
        @test_throws InexactError GPUSparseVector{Int8,Int32}(x)
        dest = GPUSparseVector(JLArray(Int32[1]), JLArray(Int8[1]), 2)
        @test_throws InexactError copyto!(dest, x)
        @test Array(dest) == Int8[1, 0]
        @test_throws InexactError (dest .= 300 .* dest)
        @test Array(dest) == Int8[1, 0]
        for vals in (Float32[1.5], Float32[Inf], ComplexF32[1+im])
            src = GPUSparseVector(JLArray(Int32[2]), JLArray(vals), 2)
            before = (Array(dest.nzInd), Array(dest.nzVal))
            @test_throws InexactError copyto!(dest, src)
            @test (Array(dest.nzInd), Array(dest.nzVal)) == before
        end
        real_dest = GPUSparseVector(JLArray(Int32[1]), JLArray(Float32[4]), 2)
        complex_src = GPUSparseVector(JLArray(Int32[2]), JLArray(ComplexF32[1+im]), 2)
        @test_throws InexactError copyto!(real_dest, complex_src)
        @test (Array(real_dest.nzInd), Array(real_dest.nzVal)) == (Int32[1], Float32[4])
    end
    @testset "ordered boolean mask slices" begin
        host = sparse(Float32[1 2 0; 3 0 4; 5 6 7])
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO)
            A = S(adapt(JLArray,host))
            for mask in (Bool[true,false,true],falses(3)), device in (false,true)
                I = device ? JLArray(mask) : mask
                B = A[I,I]
                @test Array(B) == Matrix(host[mask,mask])
                @test GPUArrays.check_structure(B) === B
                @test !Base.mightalias(A,B)
            end
            @test_throws BoundsError A[JLArray(trues(2)),JLArray(trues(3))]
        end
    end
    @testset "immutable numerical snapshots" begin
        for values in (Float32[0],Float32[-0.0],reinterpret(Float32,UInt32[0x7fc00001]))
            A=JLArray(values)
            @test GPUArrays.same_value_bits(A,copy(A))
        end
        @test !GPUArrays.same_value_bits(JLArray(Float32[0]),JLArray(Float32[-0.0]))
        a=JLArray(reinterpret(Float32,UInt32[0x7fc00001]))
        b=JLArray(reinterpret(Float32,UInt32[0x7fc00002]))
        @test !GPUArrays.same_value_bits(a,b)
        for S in (GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO)
            A=S(adapt(JLArray,sparse(Float32[2 0;1 3])))
            B=GPUArrays.snapshot_sparse_values(A)
            @test Array(A)==Array(B)
            @test !Base.mightalias(nonzeros(A),nonzeros(B))
            @test all(pair -> pair[1] === pair[2],zip(Base.front(GPUArrays.sparse_buffers(A)),Base.front(GPUArrays.sparse_buffers(B))))
        end
    end
    @testset "concatenation rejects every operand position" begin
        A = adapt(JLArray, sparse(Float32[1 0; 0 1]))
        D = JLArray(ones(Float32, 2, 2))
        for f in (hcat, vcat), pos in 1:4
            args = ntuple(i -> i == pos ? transpose(A) : D, 4)
            @test_throws ArgumentError f(args...)
            mixed = ntuple(i -> i == pos ? A : ones(Float32, 2, 2), 4)
            @test_throws ArgumentError f(mixed...)
        end
    end
end
