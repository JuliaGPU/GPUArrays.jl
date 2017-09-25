using Base.Test, GPUArrays
using GPUArrays: mapidx, gpu_sub2ind
using GPUArrays.TestSuite

function cartesian_iter(state, A, res, Asize)
    for i in CartesianRange(CartesianIndex(Asize))
        idx = gpu_sub2ind(Asize, i.I)
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
    result[1] = ntuple(Val{N}) do i
        Float32(i) * 77f0
    end
    return
end

function ntuple_closure(state, result, ::Val{N}, testval) where N
    result[1] = ntuple(Val{N}) do i
        Float32(i) * testval
    end
    return
end

function run_base(Typ)
    @testset "base functionality" begin
        @testset "mapidx" begin
            a = rand(Complex64, 77)
            b = rand(Complex64, 77)
            A = Typ(a)
            B = Typ(b)
            off = UInt32(1)
            mapidx(A, (B, off, UInt32(length(A)))) do i, a, b, off, len
                x = b[i]
                x2 = b[min(i+off, len)]
                a[i] = x * x2
            end
            foreach(1:length(a)) do i
                x = b[i]
                x2 = b[min(i+off, length(a))]
                a[i] = x * x2
            end
            @test Array(A) ≈ a
        end


        @testset "copy!" begin
            x = zeros(Float32, 10, 10)
            y = rand(Float32, 20, 10)
            a = Typ(x)
            b = Typ(y)
            r1 = CartesianRange(CartesianIndex(1, 3), CartesianIndex(7, 8))
            r2 = CartesianRange(CartesianIndex(4, 3), CartesianIndex(10, 8))
            copy!(x, r1, y, r2)
            copy!(a, r1, b, r2)
            @test x == Array(a)

            x2 = zeros(Float32, 10, 10)
            copy!(x2, r1, b, r2)
            @test x2 == x

            fill!(a, 0f0)
            copy!(a, r1, y, r2)
            @test Array(a) == x
        end
        GPUArrays.allowslow(true)
        # right now in CLArrays we fallback to geindex since on some hardware
        # somehow the vcat kernel segfaults -.-
        @testset "vcat + hcat" begin
            x = zeros(Float32, 10, 10)
            y = rand(Float32, 20, 10)
            a = Typ(x)
            b = Typ(y)
            @test vcat(x, y) == Array(vcat(a, b))
            z = rand(Float32, 10, 10)
            c = Typ(z)
            @test hcat(x, z) == Array(hcat(a, c))

            against_base(hcat, Typ{Float32}, (3, 3), (3, 3))
            against_base(vcat, Typ{Float32}, (3, 3), (3, 3))
        end
        GPUArrays.allowslow(false)

        @testset "reinterpret" begin
            a = rand(Complex64, 22)
            A = Typ(a)
            af0 = reinterpret(Float32, a)
            Af0 = reinterpret(Float32, A)
            @test Array(Af0) == af0

            a = rand(Complex64, 10 * 10)
            A = Typ(a)
            af0 = reinterpret(Float32, a, (20, 10))
            Af0 = reinterpret(Float32, A, (20, 10))
            @test Array(Af0) == af0
        end

        @testset "ntuple test" begin
            result = Typ(Vector{NTuple{3, Float32}}(1))
            gpu_call(ntuple_test, result, (result, Val{3}()))
            @test Array(result)[1] == (77, 2*77, 3*77)
            x = 88f0
            gpu_call(ntuple_closure, result, (result, Val{3}(), x))
            @test Array(result)[1] == (x, 2*x, 3*x)
        end

        @testset "cartesian iteration" begin
            Ac = rand(Float32, 32, 32)
            A = Typ(Ac)
            result = zeros(A)
            gpu_call(cartesian_iter, result, (A, result, size(A)))
            Array(result) == Ac
        end

        @testset "Custom kernel from Julia function" begin
            x = Typ(rand(Float32, 100))
            y = Typ(rand(Float32, 100))
            gpu_call(clmap!, x, (-, x, y))
            jy = Array(y)
            @test map!(-, jy, jy) ≈ Array(x)
        end
        T = Typ{Float32}
        @testset "map" begin
            against_base((a, b)-> map(+, a, b), T, (10,), (10,))
            against_base((a, b)-> map!(-, a, b), T, (10,), (10,))
            against_base((a, b, c, d)-> map!(*, a, b, c, d), T, (10,), (10,), (10,), (10,))
        end
    end
end
