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

@testset "v13 sparse regressions" begin
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
