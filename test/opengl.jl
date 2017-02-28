using GPUArrays, ModernGL
using Base.Test
import GPUArrays: GLBackend
import GLBackend: GLArray
glctx = GLBackend.init()
# using Sugar
# ast = Sugar.sugared(GLBackend.broadcast_index,
#     (GLBackend.gli.GLArray{Float32, 2}, NTuple{Int, 2}, NTuple{Int, 2}),
#     code_lowered
# )

# more complex function for broadcast
function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y*T(10.0)
end

A = GLArray(rand(Float32, 40, 40));
B = test.(A, 10f0)
decl = GLBackend.LazyMethod((GLBackend.broadcast_kernel, (GLBackend.gli.GLArray{Float32, 2}, typeof(test), GLBackend.gli.GLArray{Float32, 2}, Float32)), GLBackend.Transpiler())
println(GLBackend.getfuncsource!(decl))
for elem in GLBackend.dependencies!(decl)
    println(GLBackend.getsource!(elem))
end
using ModernGL
str = """
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
struct Test
{
    bool empty;
};
float kernel(readonly image2D x){
    return imageLoad(x, ivec2(1, 1)).x;
}
/*
float kernel(image2D x, Test arg1){
    return imageLoad(x, ivec2(1, 1)).x;
}
*/
layout (binding=0, r32f) readonly uniform image2D globalvar_A;
const Test globalvar_f = Test(false);
void main(){
    // works:
    imageLoad(globalvar_A, ivec2(1, 1)).x;
    // fails with 0:8: '' : imageLoad function cannot access the image defined
    // without layout format qualifier or with writeonly memory qualifier
    kernel(globalvar_A);
}
"""


str = """
#version 430
#extension GL_ARB_bindless_texture : enable
#extension GL_ARB_gpu_shader5 : enable
layout (local_size_x = 16, local_size_y = 16) in;
//1) Fails: needs layout qualifier for imageLoad
struct Test{
    float a;
};
vec4 func1(uint64_t handle, Test b){
    layout(rgba8) image2D a = layout(rgba8) image2D(handle);
    return imageLoad(a, ivec2(1, 1));
}
uniform uint64_t img1;
void main(){
    func1(img1, Test(1.0));
}
"""

GLAbstraction.compile_shader(Vector{UInt8}(str), GL_COMPUTE_SHADER, :test)







ast = Sugar.sugared(getindex,
    (GLBackend.gli.GLArray{Float32, 2}, NTuple{Int, 2}, NTuple{Int, 2}),
    code_lowered
)

@testset "broadcast Float32" begin
    A = GLArray(rand(Float32, 40, 40))
    A .= identity.(10.0)
    all(x-> x == 10, Array(A))
    A .= identity.(0.5)
    B = test.(A, 10.0)
    @test all(x-> isapprox(x, test(0.5f0, 10f0)), Array(B))
    A .= identity.(2.0)
    C = (*).(A, 10.0)
    @test all(x-> x == 20, Array(C))

    C = A .* 10f0
    @test all(x-> x == 20, Array(C))
    D = A .* B
    @test all(x-> x == jltest(0.5f0, 10f0) * 2, Array(D))
    D .= A .* B .+ 10f0
    @test all(x-> x == jltest(0.5f0, 10f0) * 2 + 10f0, Array(D))
end
#
# function cu_angle(z)
#     atan2(imag(z), real(z))
# end
# @testset "broadcast Complex64" begin
#     A = GLArray(fill(10f0*im, 40, 40))
#
#     A .= identity.(10f0*im)
#     @test all(x-> x == 10f0*im, Array(A))
#
#     B = cu_angle.(A)
#     @test all(x-> x == angle(10f0*im), Array(B))
#     A .= identity.(2f0*im)
#     C = A .* (2f0*im)
#     @test all(x-> x == 2f0*im * 2f0*im, Array(C))
#     D = A .* B
#     @test all(x-> x == angle(10f0*im) * 2f0*im, Array(D))
#     D .= A .* B .+ (0.5f0*im)
#     @test all(x-> x == (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
# end

# @testset "fft Complex64" begin
#     A = rand(Float32, 7,6)
#     # Move data to GPU
#     B = GLArray(A)
#     # Allocate space for the output (transformed array)
#     # Compute the FFT
#     fft!(B)
#     # Copy the result to main memory
#     # Compare against Julia's rfft
#     @test_approx_eq rfft(A) Array(B)
#     # Now compute the inverse transform
#     ifft!(B)
#     @test_approx_eq A Array(B)
# end

@testset "mapreduce" begin
    @testset "inbuilds using mapreduce (sum maximum minimum prod)" begin
        for dims in ((4048,), (1024,1024), (77,), (1923,209))
            for T in (Float32, Int32)
                A = GLArray(rand(T, dims))
                @test sum(A) ≈ sum(Array(A))
                @test maximum(A) ≈ maximum(Array(A))
                @test minimum(A) ≈ minimum(Array(A))
                @test prod(A) ≈ prod(Array(A))
            end
        end
    end
    # @testset "mapreduce with clojures" begin
    #     for dims in ((4048,), (1024,1024), (77,), (1923,209))
    #         for T in (Float32, Float64)
    #             A = GLArray(rand(T, dims))
    #             @test mapreduce(f1, op1, T(0), A) ≈ mapreduce(f1, op1, T(0), Array(A))
    #         end
    #     end
    # end
end
