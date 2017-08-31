using Base.Test
using GPUArrays
using GPUArrays: free

ctx = opencl()

# more complex function for broadcast
function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

@testset "broadcast Float32" begin
    A = GPUArray(rand(Float32, 40, 40))

    A .= identity.(10f0)
    @test all(x-> x == 10f0, Array(A))

    A .= identity.(0.5f0)
    B = test.(A, 10f0)
    @test all(x-> x ≈ test(0.5f0, 10f0), Array(B))
    A .= identity.(2f0)
    C = (*).(A, 10f0)
    @test all(x-> x == 20f0, Array(C))
    D = (*).(A, B)
    @test all(x-> x ≈ test(0.5f0, 10f0) * 2, Array(D))
    D .= (+).((*).(A, B), 10f0)
    @test all(x-> x ≈ test(0.5f0, 10f0) * 2 + 10f0, Array(D))
    free(D); free(C); free(A); free(B)
end

@testset "broadcast Complex64" begin
    A = GPUArray(fill(10f0*im, 40, 40))
    A .= identity.(10f0*im)
    @test all(x-> x == 10f0*im, Array(A))

    B = angle.(A)
    @test all(x-> x == angle(10f0*im), Array(B))
    A .= identity.(2f0*im)
    C = (*).(A, (2f0*im))
    @test all(x-> x ≈ 2f0*im * 2f0*im, Array(C))
    D = (*).(A, B)
    @test all(x-> x ≈ angle(10f0*im) * 2f0*im, Array(D))
    D .= (+).((*).(A, B), (0.5f0*im))
    @test all(x-> x ≈ (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
    free(D); free(C); free(A); free(B)
end


function clmap!(state, f, out, b)
    i = linear_index(out, state) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
    return
end
@testset "Custom kernel from Julia function" begin
    x = GPUArray(rand(Float32, 100))
    y = GPUArray(rand(Float32, 100))
    gpu_call(clmap!, x, (sin, x, y))
    # same here, x is just passed to supply a kernel size!
    jy = Array(y)
    @test map!(sin, jy, jy) ≈ Array(x)
end

@testset "Custom kernel from string function" begin
    copy_source = """
    __kernel void copy(
            __global float *dest,
            __global float *source
        ){
        int gid = get_global_id(0);
        dest[gid] = source[gid];
    }
    """
    source = GPUArray(rand(Float32, 1023, 11))
    dest = GPUArray(zeros(Float32, size(source)))
    f = (copy_source, :copy)
    gpu_call(f, dest, (dest, source))
    @test Array(dest) == Array(source)
end


@testset "transpose" begin
    A = rand(Float32, 32, 32)
    Agpu = GPUArray(A)
    @test Array(Agpu') == A'
end
