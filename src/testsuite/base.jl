function cartesian_iter(state, A, res, Asize)
    for i in CartesianIndices(Asize)
        idx = GPUArrays.gpu_sub2ind(Asize, i.I)
        res[idx] = A[idx]
    end
    return
end

function clmap!(state, f, out, b)
    i = linear_index(state) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
    return
end

function ntuple_test(state, result, ::Val{N}) where N
    result[1] = ntuple(Val(N)) do i
        Float32(i) * 77f0
    end
    return
end

function ntuple_closure(state, result, ::Val{N}, testval) where N
    result[1] = ntuple(Val(N)) do i
        Float32(i) * testval
    end
    return
end

function test_base(AT)
    @testset "base functionality" begin
        @testset "copyto!" begin
            x = fill(0f0, (10, 10))
            y = rand(Float32, (20, 10))
            a = AT(x)
            b = AT(y)
            r1 = CartesianIndices((1:7, 3:8))
            r2 = CartesianIndices((4:10, 3:8))
            copyto!(x, r1, y, r2)
            copyto!(a, r1, b, r2)
            @test x == Array(a)

            x2 = fill(0f0, (10, 10))
            copyto!(x2, r1, b, r2)
            @test x2 == x

            fill!(a, 0f0)
            copyto!(a, r1, y, r2)
            @test Array(a) == x
        end

        @testset "vcat + hcat" begin
            @test compare(vcat, AT, fill(0f0, (10, 10)), rand(Float32, 20, 10))
            @test compare(hcat, AT, fill(0f0, (10, 10)), rand(Float32, 10, 10))

            @test compare(hcat, AT, rand(Float32, 3, 3), rand(Float32, 3, 3))
            @test compare(vcat, AT, rand(Float32, 3, 3), rand(Float32, 3, 3))
            @test compare((a,b) -> cat(a, b; dims=4), AT, rand(Float32, 3, 4), rand(Float32, 3, 4))
        end

        @testset "reinterpret" begin
            a = rand(ComplexF32, 22)
            A = AT(a)
            af0 = reinterpret(Float32, a)
            Af0 = reinterpret(Float32, A)
            @test Array(Af0) == af0

            a = rand(ComplexF32, 10 * 10)
            A = AT(a)
            af0 = reshape(reinterpret(Float32, vec(a)), (20, 10))
            Af0 = reshape(reinterpret(Float32, vec(A)), (20, 10))
            @test Array(Af0) == af0
        end

        @testset "ntuple test" begin
            result = AT(Vector{NTuple{3, Float32}}(undef, 1))
            gpu_call(ntuple_test, result, (result, Val(3)))
            @test Array(result)[1] == (77, 2*77, 3*77)
            x = 88f0
            gpu_call(ntuple_closure, result, (result, Val(3), x))
            @test Array(result)[1] == (x, 2*x, 3*x)
        end

        @testset "cartesian iteration" begin
            Ac = rand(Float32, 32, 32)
            A = AT(Ac)
            result = fill!(copy(A), 0.0)
            gpu_call(cartesian_iter, result, (A, result, size(A)))
            Array(result) == Ac
        end

        @testset "Custom kernel from Julia function" begin
            x = AT(rand(Float32, 100))
            y = AT(rand(Float32, 100))
            gpu_call(clmap!, x, (-, x, y))
            jy = Array(y)
            @test map!(-, jy, jy) ≈ Array(x)
        end

        @testset "map" begin
            @test compare((a, b)-> map(+, a, b),    AT, rand(Float32, 10), rand(Float32, 10))
            @test compare((a, b)-> map!(-, a, b),   AT, rand(Float32, 10), rand(Float32, 10))
            @test compare((a, b, c, d)-> map!(*, a, b, c, d), AT, rand(Float32, 10), rand(Float32, 10), rand(Float32, 10), rand(Float32, 10))
        end

        @testset "repeat" begin
            @test compare(a-> repeat(a, 5, 6),  AT, rand(Float32, 10))
            @test compare(a-> repeat(a, 5),     AT, rand(Float32, 10))
            @test compare(a-> repeat(a, 5),     AT, rand(Float32, 5, 4))
            @test compare(a-> repeat(a, 4, 3),  AT, rand(Float32, 10, 15))
        end

        @testset "heuristics" begin
            blocks, threads = thread_blocks_heuristic(0)
            @test blocks == (1,)
            @test threads == (1,)
        end
    end
end
