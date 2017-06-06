using GPUArrays

w = GLBackend.init()

A = GPUArray(rand(Float32, 1024));
B = GPUArray(rand(Float32, 1024));
function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

x = test.(A, B)
