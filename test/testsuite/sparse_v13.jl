# Shared semantic gates for reference and physical backends.

@kernel function sparse_lookup_v13_kernel(out, A)
    i = @index(Global, Linear)
    out[i] = A[i, i]
end

# Affine-map composition modulo a prime: exactly associative, noncommutative,
# represented by an isbits integer that every backend can transfer.
function compose_affine_v13(x::UInt32, y::UInt32)
    a, b = x & 0xffff, x >> 16
    c, d = y & 0xffff, y >> 16
    ((a * c) % 251) | (((a * d + b) % 251) << 16)
end

function sparse_v13(AT, eltypes)
    @testset "v13 contracts" begin
        @testset "$S" for S in sparse_matrix_formats
            host = sparse(Float32[1 0 0; 2 0 0; 0 0 3])
            A = gpu_sparse(AT, S, host)
            v = gpu_sparse(AT, sparsevec([1,3],Float32[2,4],3))
            dense_v = AT(Float32[2,3,4])
            before = copy(A)
            @test_throws ErrorException A .* v
            @test_throws ErrorException v .* A
            @test_throws ErrorException 2f0 .* A .+ v
            @test_throws ErrorException A .* v .+ dense_v
            @test_throws ErrorException dense_v .+ v .* A
            scalar = AT(fill(0f0))
            @test_throws r"both sparse vectors and sparse matrices" (A .* v) .+ scalar
            @test_throws r"both sparse vectors and sparse matrices" scalar .+ (A .* v)
            @test_throws r"both sparse vectors and sparse matrices" A .* (scalar .+ v)
            @test_throws r"both sparse vectors and sparse matrices" (v .+ scalar) .* A
            @test_throws r"zero-dimensional GPU arrays" A .+ scalar
            @test_throws r"zero-dimensional GPU arrays" scalar .+ A
            @test_throws r"zero-dimensional GPU arrays" A .+ view(dense_v,1)
            @test_throws r"zero-dimensional GPU arrays" (A .= A .+ scalar)
            @test_throws r"zero-dimensional GPU arrays" v .+ scalar
            @test_throws r"zero-dimensional GPU arrays" scalar .+ v
            @test_throws ErrorException (A .= A .* v)
            @test A == before
            @test Array(A .* dense_v) == Array(host) .* Float32[2,3,4]
            @test Array(dense_v .+ A) == Float32[2,3,4] .+ Array(host)
            @test Array(A .+ AT(Float32[2])) == Array(host) .+ 2f0
            row = gpu_sparse(AT,S,sparse(reshape(Float32[1,0,2],1,3)))
            @test_throws ArgumentError row .+ AT(Float32[2,3])
            @test_throws ArgumentError AT(Float32[2,3]) .* row
            column = gpu_sparse(AT,S,sparse(reshape(Float32[1,0,2],3,1)))
            @test_throws ArgumentError column .+ AT(ones(Float32,3,2))
            singleton = gpu_sparse(AT,sparsevec([1],Float32[2],1))
            @test_throws ArgumentError singleton .+ v
            out = AT(zeros(Float32, 3))
            sparse_lookup_v13_kernel(get_backend(out))(out, A; ndrange=3)
            @test Array(out) == diag(host)
            @test A == copy(A)
            B = copy(A)
            B .*= 2
            @test Array(B) == 2 .* Array(host)
            B .= B .+ A
            @test Array(B) == 3 .* Array(host)
            left = sparse(Float32[1 2 3; 0 0 0; 0 0 0])
            right = sparse(Float32[0 0 0; 4 0 5; 0 6 0])
            fused = 2f0 .* gpu_sparse(AT,S,left) .+ 3f0 .* gpu_sparse(AT,S,right)
            @test Array(GPUArrays.check_structure(fused)) == 2f0 .* Array(left) .+ 3f0 .* Array(right)
            @test Array(A[:]) == vec(Array(host))
            @test nnz(transpose(A)) == nnz(A)
            @test sum(transpose(A)) == sum(host)
            @test Array(sum(adjoint(A); dims=1)) == sum(adjoint(host); dims=1)
            @test_throws ArgumentError hcat(A, A)
            @test_throws ArgumentError vcat(A, A)
            before = copy(A)
            @test_throws ArgumentError (A .= A .+ 1)
            @test A == before
            y = AT(fill(Float32(NaN), 3)); x = AT(ones(Float32, 3))
            GPUArrays.spmv!(y, 'N', A, x, 1f0, 0f0; algorithm=GPUArrays.Portable())
            @test Array(y) == host * ones(Float32, 3)
            candidate = GPUArrays.vendor_spmv(y, 'N', A, x, 1f0, 0f0)
            if candidate isa GPUArrays.UnsupportedSparseVendor
                @test_throws ArgumentError GPUArrays.spmv!(y, 'N', A, x, 1f0, 0f0;
                                                           algorithm=GPUArrays.Vendor())
            else
                GPUArrays.spmv!(y, 'N', A, x, 1f0, 0f0; algorithm=GPUArrays.Vendor())
            end
            @test Array(y) == host * ones(Float32, 3)
            for wrapper in (Symmetric, Hermitian), uplo in (:U, :L)
                host = sparse(Float32[1 2; 4 3])
                A = gpu_sparse(AT, S, host)
                @test Array(wrapper(A, uplo) * AT(Float32[Inf, 1])) ==
                      wrapper(host, uplo) * Float32[Inf, 1]
            end
        end
        @testset "mixed pointer and coordinate storage" begin
            for S in (GPUSparseMatrixCSR, GPUSparseMatrixCSC)
                A = S(AT(Int64[1, 3, 4]), AT(Int32[1, 2, 2]), AT(Float32[1, 2, 3]), (2, 2))
                for B in (copy(A), similar(A), similar(A, Float32), S{Float32}(A), zero(A))
                    @test eltype(GPUArrays.compressed_ptr(B)) == Int64
                    @test SparseArrays.indtype(B) == Int32
                    @test !Base.mightalias(A, B)
                end
                B = A .+ A
                @test eltype(GPUArrays.compressed_ptr(B)) == Int64
                @test SparseArrays.indtype(B) == Int32
                @test Array(B) == 2 .* Array(A)
                @test Array(GPUArrays.check_structure(B)) == Array(B)
                backing = AT(Float32[1, 9, 2, 9, 3, 9])
                bad = S(copy(GPUArrays.compressed_ptr(A)), copy(GPUArrays.compressed_ind(A)),
                        view(backing, 1:2:6), size(A))
                before = map(Array, GPUArrays.sparse_buffers(bad))
                @test_throws ArgumentError copyto!(bad, zero(A))
                @test map(Array, GPUArrays.sparse_buffers(bad)) == before
                for Tp in (Int32,Int64)
                    shape = S === GPUSparseMatrixCSR ? (1,4) : (4,1)
                    four = S(AT(Tp[1,5]),AT(Int32[1,2,3,4]),AT(Float32[1,2,3,4]),shape)
                    empty = zero(four)
                    merged = 2f0 .* four .+ 3f0 .* empty
                    @test nnz(merged) == 4
                    @test Array(GPUArrays.check_structure(merged)) == 2f0 .* Array(four)
                end
            end
        end
        @testset "ordered duplicate assembly" begin
            I = AT(Int32[2, 1, 2, 1, 2, 1])
            V = AT(UInt32[2 | (3 << 16), 4 | (5 << 16), 6 | (7 << 16),
                          8 | (9 << 16), 10 | (11 << 16), 12 | (13 << 16)])
            x = sparsevec(I, V, 3, compose_affine_v13)
            expected = sparsevec(Array(I), Array(V), 3, compose_affine_v13)
            @test Array(x) == Array(expected)
            # Packed-key overflow has a tiny lexicographic test independent of allocation.
            keys = GPUArrays.packed_keys(AT(Int64[2, 1, 2]), AT(Int64[1, 2, 1]),
                                         (typemax(Int), 2))
            @test Array(sortperm(keys; alg=Base.Sort.MergeSort)) == [2, 1, 3]
        end
    end
end
